import math
import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from build_sam_new import sam_model_registry_new
from dataset import SingleJSONDataset
from utils.transforms import ResizeLongestSide


def calculate_accuracy(pred_mask, gt_mask):
    pred_mask=pred_mask.bool()
    gt_mask = gt_mask.bool()
    batch_size = pred_mask.size(0)
    batch_accuracies = []
    for i in range(batch_size):
        pred_batch = pred_mask[i]
        gt_batch = gt_mask[i]
        correct = (pred_batch == gt_batch).sum().item()
        total = gt_batch.numel()
        accuracy = correct / total
        batch_accuracies.append(accuracy)
    # 计算平均值
    average_accuracy = sum(batch_accuracies) / len(batch_accuracies)
    return average_accuracy


def calculate_iou(pred_mask, gt_mask):
    batch_size = pred_mask.size(0)
    batch_ious = []
    for i in range(batch_size):
        pred = pred_mask[i].bool()
        gt = gt_mask[i].bool()
        intersection = (pred & gt).sum().item()
        union = (pred | gt).sum().item()
        if union == 0:
            iou = 0
        else:
            iou = intersection / union
        batch_ious.append(iou)
    average_iou = sum(batch_ious) / len(batch_ious)
    return average_iou

# 定义 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        batch_losses = []
        for i in range(batch_size):
            input_batch = inputs[i].unsqueeze(0)
            target_batch = targets[i].unsqueeze(0)
            BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(input_batch, target_batch)
            pt = torch.exp(-BCE_loss)
            F_loss_batch = self.alpha * (1 - pt) ** self.gamma * BCE_loss
            batch_losses.append(F_loss_batch.mean())
        return torch.stack(batch_losses).mean()

# 定义 Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        smooth = 1e-5
        inputs = torch.sigmoid(inputs)  # 将输入转换为概率值
        # 对每个样本单独计算交集
        intersection = (inputs * targets).sum(dim=(1, 2))
        # 对每个样本单独计算输入和目标的和
        inputs_sum = inputs.sum(dim=(1, 2))
        targets_sum = targets.sum(dim=(1, 2))
        dice = (2. * intersection + smooth) / (inputs_sum + targets_sum + smooth)
        return 1 - dice.mean()

def resize_longest_side(image, target_length):
    # 获取图像的原始宽度和高度
    width, height = image.size
    # 判断较长边
    if width > height:
        # 如果宽度是较长边，计算新的高度
        new_width = target_length
        new_height = int(height * (target_length / width))
    else:
        # 如果高度是较长边，计算新的宽度
        new_height = target_length
        new_width = int(width * (target_length / height))
    # 调整图像大小
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    return resized_image

def postprocess_masks(
        masks: torch.Tensor,  # 从掩码解码器输出的批量掩码，形状为 (B, C, H, W)，B 是批量大小，C 是掩码通道数，H 和 W 是掩码的高度和宽度
        img_size,
        input_size: Tuple[int, ...],  # 输入到模型的图像尺寸，格式为 (H, W)，用于去除填充
        original_size: Tuple[int, ...],  # 在调整大小以输入到模型之前，图像的原始尺寸，格式为 (H, W)
) -> torch.Tensor:
    """
    Remove padding and upscale masks to the original image size.
    移除填充并将掩码上采样到原始图像的尺寸。
    Arguments:
      masks (torch.Tensor): Batched masks from the mask_decoder,
        in BxCxHxW format.
      input_size (tuple(int, int)): The size of the image input to the
        model, in (H, W) format. Used to remove padding.
      original_size (tuple(int, int)): The original size of the image
        before resizing for input to the model, in (H, W) format.

    Returns:
      (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
        is given by original_size.# 批量掩码，格式为 BxCxHxW，其中 (H, W) 是原始图像的尺寸
    """
    # 将掩码上采样到图像编码器期望的输入尺寸（通常是一个固定的正方形尺寸）
    # 使用双线性插值方法进行上采样，align_corners=False 表示不将角点对齐
    masks = F.interpolate(
        masks,
        (img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )
    # 去除之前为了将图像调整为正方形而添加的填充部分
    # 只保留与输入图像实际尺寸对应的部分
    masks = masks[..., : input_size[0], : input_size[1]]
    # 将去除填充后的掩码上采样到原始图像的尺寸
    # 同样使用双线性插值方法，align_corners=False 表示不将角点对齐
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks  # 返回处理后的批量掩码


def preprocess(x: torch.Tensor,img_size) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    pixel_mean = [123.675, 116.28, 103.53],  # 用于对输入图像像素进行归一化的均值，默认值适用于常见的图像数据集
    pixel_std = [58.395, 57.12, 57.375],  # 用于对输入图像像素进行归一化的标准差，默认值适用于常见的图像数据集
    # 将像素均值、标准差注册为缓冲区，以便在模型保存和加载时一起保存和加载
    pixel_mean=torch.Tensor(pixel_mean).view(-1, 1, 1)  # 并将其形状调整为 (通道数, 1, 1) 以便于广播
    pixel_std=torch.Tensor(pixel_std).view(-1, 1, 1) # 并将其形状调整为 (通道数, 1, 1) 以便于广播

    # 对像素值进行归一化处理，并将图像填充为正方形输入
    # Normalize colors
    x = (x - pixel_mean) / pixel_std  # 对图像的像素值进行归一化，减去像素均值并除以像素标准差

    # Pad
    h, w = x.shape[-2:]  # 获取图像的高度和宽度
    padh = img_size - h  # 计算在高度方向上需要填充的像素数，以达到图像编码器期望的输入尺寸
    padw = img_size - w  # 计算在宽度方向上需要填充的像素数，以达到图像编码器期望的输入尺寸
    # 使用 PyTorch 的 F.pad 函数对图像进行填充
    # 填充顺序为 (左, 右, 上, 下)，这里只在右侧和底部进行填充
    x = F.pad(x, (0, padw, 0, padh))
    return x  # 返回预处理后的图像

def init_model(image_encoder_pth=None):
    # 初始化 SAM 预处理变换
    sam_transform = ResizeLongestSide(1024)

    model_type = "pvt_small"
    # 加载预训练权重文件
    checkpoint_path = "sam_vit_b_01ec64.pth"

    # 加载SAM模型
    sam = sam_model_registry_new[model_type]()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 提取提示编码器和掩码解码器的权重
    #image_encoder_weights = {k.replace('image_encoder.', ''): v for k, v in checkpoint.items() if k.startswith('image_encoder.')}
    prompt_encoder_weights = {k.replace('prompt_encoder.', ''): v for k, v in checkpoint.items() if k.startswith('prompt_encoder.')}
    mask_decoder_weights = {k.replace('mask_decoder.', ''): v for k, v in checkpoint.items() if k.startswith('mask_decoder.')}

    # 加载提示编码器和掩码解码器的权重
    #sam.image_encoder.load_state_dict(image_encoder_weights)
    sam.prompt_encoder.load_state_dict(prompt_encoder_weights)
    sam.mask_decoder.load_state_dict(mask_decoder_weights)

    if image_encoder_pth is not None:
        # 加载保存的模型参数
        state_dict = torch.load(image_encoder_pth, map_location=device)
        # 将加载的参数应用到模型中
        sam.image_encoder.load_state_dict(state_dict)

    # 冻结提示编码器和掩码解码器的参数
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    for param in sam.mask_decoder.parameters():
        param.requires_grad = False
    return sam


import cv2
def process_single(sam, image_file_path, point_tensor):
    transform = ResizeLongestSide(sam.image_encoder.img_size)  # 创建一个 ResizeLongestSide 实例，用于将图像的最长边调整为模型期望的尺寸
    # 读取图像
    image = cv2.imread(image_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    assert "RGB" in [  # 检查图像格式是否为 'RGB' 或 'BGR'，如果不是则抛出异常
        "RGB",
        "BGR",
    ], f"image_format must be in ['RGB', 'BGR'], is RGB."
    if "RGB" != "RGB":  # 如果图像格式与模型期望的格式不一致，则反转通道顺序
        image = image[..., ::-1]

    input_image = transform.apply_image(image)  # 应用 ResizeLongestSide 变换，将图像的最长边调整为模型期望的尺寸
    input_image_torch = torch.as_tensor(input_image, device=sam.device)  # 将 NumPy 数组转换为 PyTorch 张量，并将其放置在与模型相同的设备上
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :,:]  # 调整张量的维度顺序为 BCHW（批次、通道、高度、宽度），并确保内存连续

    original_size = image.shape[:2]  # 保存原始图像的尺寸
    input_size = tuple(input_image_torch.shape[-2:])  # 保存输入图像（变换后）的尺寸
    input_image = sam.preprocess(input_image_torch)  # 对输入图像进行预处理，如归一化、填充等操作
    image_embeddings = sam.image_encoder(input_image)
    """
    image = Image.open(image_file_path).convert('RGB')
    original_width, original_height = image.size  ##
    image = resize_longest_side(image, 1024)
    input_width, input_height = image.size  ##

    # 应用转换操作
    to_tensor = transforms.ToTensor()
    image_tensor =  to_tensor(image).to(sam.device) # 3*h*w
    image_tensor=sam.preprocess(image_tensor)
    image_tensor=image_tensor.unsqueeze(0)  # 1*3*1024*1024

    # 前向传播
    image_embeddings = sam.image_encoder(image_tensor)
    """

    point_tensor = torch.tensor(transform.apply_coords(point_tensor.numpy(), original_size),device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配
    # 创建形状为 (b, n) 且内容全为 1 的向量
    label_tensor = torch.ones((point_tensor.shape[0], point_tensor.shape[1]), dtype=torch.float32,device=sam.device)  # b*n
    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=(point_tensor, label_tensor),
        boxes=None,
        masks=None
    )
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False  ##
    )
    masks = sam.postprocess_masks(  # 对低分辨率掩码进行后处理，将其调整到输入图像的原始尺寸
        low_res_masks,
        input_size=input_size,  ##
        original_size=original_size,
    )

    return masks

def train(dataset_folder_path,epochs,batch_size,save_Path):
    sam=init_model()
    #transform = ResizeLongestSide(sam.image_encoder.img_size)  # 创建一个 ResizeLongestSide 实例，用于将图像的最长边调整为模型期望的尺寸
    optimizer = optim.Adam(sam.image_encoder.parameters(), lr=1e-3)
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    save_dir=save_Path
    best_iou=0
    best_accuracy=0
    best_loss = 1000
    best_epoch=0

    all_image_files = [f for f in os.listdir(dataset_folder_path) if f.endswith('.jpg')][:100]
    num_files = len(all_image_files)
    num_train = int(num_files * 0.7)
    num_val = int(num_files * 0.15)
    num_test = num_files - num_train - num_val

    # 划分数据集
    train_image_files = all_image_files[:num_train]
    val_image_files = all_image_files[num_train:num_train + num_val]
    test_image_files = all_image_files[num_train + num_val:]

    for epoch in range(epochs):
        # 训练阶段
        sam.train()
        #epoch_loss = 0
        # 遍历文件夹中的所有文件
        for index, image_filename in enumerate(train_image_files, start=1):
            json_filename = os.path.splitext(image_filename)[0] + '.json'
            json_file_path = os.path.join(dataset_folder_path, json_filename)
            image_file_path = os.path.join(dataset_folder_path, image_filename)

            # 检查对应的JSON文件是否存在
            if os.path.exists(json_file_path):
                dataset = SingleJSONDataset(json_file_path)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                total_batch=math.ceil(dataset.count / batch_size)
                for batch_idx, (point_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):#point_tensor:b12    gt_mask_tensor:bhw
                    masks = process_single(sam, image_file_path, point_tensor).cpu()#bchw
                    #masks = masks > 0.0  # 根据掩码阈值将掩码转换为二进制掩码

                    # 计算损失
                    focal = focal_loss(masks.squeeze(1) , gt_mask_tensor)
                    dice = dice_loss(masks.squeeze(1) , gt_mask_tensor)
                    loss = 20 * focal + dice

                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    accuracy = calculate_accuracy(masks.squeeze(1)> 0.0, gt_mask_tensor)
                    iou = calculate_iou(masks.squeeze(1)> 0.0, gt_mask_tensor)

                    #epoch_loss += loss.item()
                    print(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, file:{index}/{num_train}  batch:{batch_idx}/{total_batch}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')

        # 验证阶段
        sam.eval()
        total_val_loss = 0
        total_accuracy = 0
        total_iou = 0
        total_num=0
        with torch.no_grad():
            # 遍历文件夹中的所有文件
            for index, image_filename in enumerate(val_image_files, start=1):
                json_filename = os.path.splitext(image_filename)[0] + '.json'
                json_file_path = os.path.join(dataset_folder_path, json_filename)
                image_file_path = os.path.join(dataset_folder_path, image_filename)

                # 检查对应的JSON文件是否存在
                if os.path.exists(json_file_path):
                    dataset = SingleJSONDataset(json_file_path)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    total_batch = math.ceil(dataset.count / batch_size)
                    total_num += total_batch
                    for batch_idx, (point_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):
                        masks = process_single(sam, image_file_path, point_tensor).cpu()
                        # 计算损失
                        focal = focal_loss(masks.squeeze(1), gt_mask_tensor)
                        dice = dice_loss(masks.squeeze(1), gt_mask_tensor)
                        loss = 20 * focal + dice
                        total_val_loss += loss.item()

                        masks = masks > 0.0
                        accuracy = calculate_accuracy(masks.squeeze(1)> 0.0, gt_mask_tensor)
                        total_accuracy += accuracy

                        iou = calculate_iou(masks.squeeze(1)> 0.0, gt_mask_tensor)
                        total_iou += iou
                        print(f'VAL ==> Epoch {epoch + 1}/{epochs}, file:{index}/{num_val}  batch:{batch_idx}/{total_batch}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')

        avg_val_loss = total_val_loss / total_num
        avg_accuracy = total_accuracy / total_num
        avg_iou = total_iou / total_num
        print(f'VAL ==> Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')

        # 保存图像编码器的参数
        checkpoint_path = os.path.join(save_dir, f'image_encoder_epoch_{epoch + 1}.pth')
        torch.save(sam.image_encoder.state_dict(), checkpoint_path)
        print(f'Saved image encoder parameters to {checkpoint_path}')

        if avg_val_loss<best_loss and avg_accuracy>best_accuracy and avg_iou>best_iou:
            best_epoch=epoch + 1
            best_loss=avg_val_loss
            best_accuracy=avg_accuracy
            best_iou=avg_iou
            # 保存图像编码器的参数
            checkpoint_path = os.path.join(save_dir, f'image_encoder_best.pth')
            torch.save(sam.image_encoder.state_dict(), checkpoint_path)

    print(f'训练已完成 Epoch：{epochs}, Batch: {batch_size}, num_train: {num_train}, num_val: {num_val}, best_epoch:{best_epoch}, best_loss:{best_loss}, best_accuracy:{best_accuracy}, best_iou:{best_iou} ')

    # 测试阶段
    test_sam=init_model(os.path.join(save_dir, f'image_encoder_best.pth'))
    test_sam.eval()
    total_test_loss = 0
    total_accuracy = 0
    total_iou = 0
    total_num = 0
    with torch.no_grad():
        # 遍历文件夹中的所有文件
        for index, image_filename in enumerate(test_image_files, start=1):
            json_filename = os.path.splitext(image_filename)[0] + '.json'
            json_file_path = os.path.join(dataset_folder_path, json_filename)
            image_file_path = os.path.join(dataset_folder_path, image_filename)

            # 检查对应的JSON文件是否存在
            if os.path.exists(json_file_path):
                dataset = SingleJSONDataset(json_file_path)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                total_batch = math.ceil(dataset.count / batch_size)
                total_num += total_batch
                for batch_idx, (point_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):
                    masks = process_single(test_sam, image_file_path, point_tensor).cpu()
                    # 计算损失
                    focal = focal_loss(masks.squeeze(1), gt_mask_tensor)
                    dice = dice_loss(masks.squeeze(1), gt_mask_tensor)
                    loss = 20 * focal + dice
                    total_test_loss += loss.item()

                    masks = masks > 0.0
                    accuracy = calculate_accuracy(masks.squeeze(1)> 0.0, gt_mask_tensor)
                    total_accuracy += accuracy

                    iou = calculate_iou(masks.squeeze(1)> 0.0, gt_mask_tensor)
                    total_iou += iou
                    print(f'TEST ==> file:{index}/{num_test}  batch:{batch_idx}/{total_batch}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')

    avg_test_loss = total_test_loss / total_num
    avg_accuracy = total_accuracy / total_num
    avg_iou = total_iou / total_num
    print(f'TEST ==> Test Loss: {avg_test_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')

#def test(dataset_folder_path,epochs=10,batch_size=16):

# 保存训练好的图像编码器的参数
#torch.save(sam.image_encoder.state_dict(), 'trained_image_encoder.pth')
datasetPath='C:\\Users\\yangxinyao\\Downloads\\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0'
save_Path='C:\\Users\\yangxinyao\\Downloads\\checkpoint2'
train(datasetPath,5,16,save_Path)

