import logging
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
from tqdm import tqdm

from build_sam import sam_model_registry
from dataset import SingleJSONDataset
from utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import gaussian_blur

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
    """
        def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
    """


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


class DBSLoss(nn.Module):
    def __init__(self, sigma=5, smooth=1e-5):
        super(DBSLoss, self).__init__()
        self.sigma = sigma
        self.smooth = smooth

    def _generate_weights(self, mask):
        # 扩展维度以适应卷积操作
        mask = mask.unsqueeze(1)
        # 定义边缘检测卷积核
        kernel = torch.tensor([[[
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]]], dtype=torch.float32).to(mask.device)

        # 进行卷积操作
        boundary = F.conv2d(mask, kernel, padding=1)

        # 二值化处理得到边界掩码
        boundary_mask = (boundary > 0).float()

        # 高斯模糊生成权重图（边界附近权重更高）
        # 计算高斯核的大小，确保覆盖 3σ 范围
        kernel_size = int(self.sigma * 4 + 1)
        # 调用 gaussian_blur 函数对边界图进行高斯模糊处理
        weight_map = gaussian_blur(
            boundary_mask,
            kernel_size=kernel_size,
            sigma=self.sigma
        )  # (N, 1, H, W)

        # 归一化到 [0, 1] 并去除通道维度
        # 找到每个样本的最大权重值，先在高度方向上取最大值，再在宽度方向上取最大值
        max_values = weight_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        # 为了避免除零错误，加上一个很小的数 1e-6
        weight_map = weight_map / (max_values + 1e-6) + 1.0
        # 去除通道维度，得到 (N, H, W) 形状的权重图
        return weight_map.squeeze(1)  # (N, H, W)

    def forward(self, inputs, targets):
        smooth = self.smooth
        inputs = torch.sigmoid(inputs)  # 将输入转换为概率值

        # 生成边界权重图
        weights = self._generate_weights(targets)  # (N, H, W)

        # 对每个样本单独计算加权交集
        intersection = (inputs * targets * weights).sum(dim=(1, 2))
        # 对每个样本单独计算加权输入和目标的和
        inputs_sum = (inputs * weights).sum(dim=(1, 2))
        targets_sum = (targets * weights).sum(dim=(1, 2))

        dice = (2. * intersection + smooth) / (inputs_sum + targets_sum + smooth)
        return 1 - dice.mean()


def init_model(checkpoint_path=None):
    model_type = "vit_b"
    sam_checkpoint_path = "sam_vit_b_01ec64.pth"
    # 获取当前脚本文件所在的目录
    #current_dir = os.path.dirname(os.path.abspath(__file__))
    #checkpoint_path = os.path.join(current_dir, "checkpoint_bb","sam_epoch_3.pth")

    # 加载SAM模型
    sam = sam_model_registry[model_type]()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device)
    
    if checkpoint_path is None:
        checkpoint = torch.load(sam_checkpoint_path, map_location=device)
        
        log = sam.load_state_dict(checkpoint, strict=False)

        print("Model loaded from {} \n => {}".format(sam_checkpoint_path, log))


        sam.prompt_encoder.rbox_point_embeddings[0].weight.data = sam.prompt_encoder.point_embeddings[2].weight.data.clone()
        sam.prompt_encoder.rbox_point_embeddings[1].weight.data = sam.prompt_encoder.point_embeddings[3].weight.data.clone()
        sam.prompt_encoder.rbox_point_embeddings[2].weight.data = sam.prompt_encoder.point_embeddings[3].weight.data.clone()
        sam.prompt_encoder.rbox_point_embeddings[3].weight.data = sam.prompt_encoder.point_embeddings[2].weight.data.clone()
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        log = sam.load_state_dict(checkpoint, strict=False)

        print("Model loaded from {} \n => {}".format(checkpoint_path, log))


    # 提取提示编码器和掩码解码器的权重
    #image_encoder_weights = {k.replace('image_encoder.', ''): v for k, v in checkpoint.items() if k.startswith('image_encoder.')}
    #prompt_encoder_weights = {k.replace('prompt_encoder.', ''): v for k, v in checkpoint.items() if k.startswith('prompt_encoder.')}
    #mask_decoder_weights = {k.replace('mask_decoder.', ''): v for k, v in checkpoint.items() if k.startswith('mask_decoder.')}

    # 加载提示编码器和掩码解码器的权重
    #sam.image_encoder.load_state_dict(image_encoder_weights)
    #sam.prompt_encoder.load_state_dict(prompt_encoder_weights)
    #sam.mask_decoder.load_state_dict(mask_decoder_weights)

   

    # 冻结原有参数
    for name, param in sam.named_parameters():
        if 'rbox_point_embeddings' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            print(f"可训练参数名: {name}")


    """
    if image_encoder_pth is not None:
        # 加载保存的模型参数
        state_dict = torch.load(image_encoder_pth, map_location=device)
        # 将加载的参数应用到模型中
        sam.image_encoder.load_state_dict(state_dict)

    # 冻结图像编码器的参数
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.mask_decoder.parameters():
        param.requires_grad = False
    """


    """
    # 冻结提示编码器和掩码解码器的参数
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    """

    return sam


import cv2
def process_single(sam, image_file_path, rbox_tensor):
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
    


    rbox_tensor = torch.tensor(transform.apply_rboxes(rbox_tensor.numpy(), original_size),device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        boxes=None,
        rboxes=rbox_tensor,
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

from dataset_coco import COCODataset_AllAnnotations,COCODataset_OneImage
def train(epochs,batch_size,save_Path):
    logging.basicConfig(filename='training-44.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    #checkpoint_path=r'./checkpoint_new/sam_best_lr4_2000_new.pth'
    checkpoint_path=None
    sam=init_model(checkpoint_path)
    # 定义优化器，只优化可训练的参数
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sam.parameters()), lr=1e-4)

    #transform = ResizeLongestSide(sam.image_encoder.img_size)  # 创建一个 ResizeLongestSide 实例，用于将图像的最长边调整为模型期望的尺寸
    #optimizer = optim.Adam(sam.image_encoder.parameters(), lr=1e-3)
    #optimizer = optim.Adam(sam.mask_decoder.parameters(), lr=1e-4)

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    #dice_loss = DBSLoss()
    save_dir=save_Path
    best_iou=0
    best_accuracy=0
    best_loss = 1000
    best_epoch=0


    num_train = 1000
    num_val = 100

    train_ann_file = r'/root/autodl-tmp/coco2017/annotations/instances_train2017_with_rbox.json'
    train_image_dir = r'/root/autodl-tmp/coco2017/train2017'
    val_ann_file = r'/root/autodl-tmp/coco2017/annotations/instances_val2017_with_rbox.json'
    val_image_dir = r'/root/autodl-tmp/coco2017/val2017'
    train_image_dataset = COCODataset_AllAnnotations(train_ann_file, train_image_dir,max_data_size=num_train)
    train_image_dataloader = DataLoader(train_image_dataset, batch_size=1)
    val_image_dataset = COCODataset_AllAnnotations(val_ann_file, val_image_dir, max_data_size=num_val)
    val_image_dataloader = DataLoader(val_image_dataset, batch_size=1)

    for epoch in range(epochs):
        # 训练阶段
        sam.train()
        total_val_loss = 0
        total_accuracy = 0
        total_iou = 0
        total_num = 0
        for image_index,(img_path, annotations) in tqdm(enumerate(train_image_dataloader, start=1), total=len(train_image_dataloader), desc=f'Training ==> Epoch:{epoch + 1}/{epochs}'):
            oneImage_dataset = COCODataset_OneImage(annotations)
            oneImage_dataloader = DataLoader(oneImage_dataset, batch_size=batch_size)
            total_num += len(oneImage_dataloader)
            for batch_idx, (rbox_tensor, gt_mask_tensor) in enumerate(oneImage_dataloader,start=1):  # point_tensor:b12    gt_mask_tensor:bhw
                masks = process_single(sam, img_path[0], rbox_tensor).cpu()#bchw
                #masks = masks > 0.0  # 根据掩码阈值将掩码转换为二进制掩码

                # 计算损失
                focal = focal_loss(masks.squeeze(1) , gt_mask_tensor)
                dice = dice_loss(masks.squeeze(1) , gt_mask_tensor)
                loss = 20 * focal + dice
                total_val_loss += loss.item()

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accuracy = calculate_accuracy(masks.squeeze(1)> 0.0, gt_mask_tensor)
                total_accuracy += accuracy
                iou = calculate_iou(masks.squeeze(1)> 0.0, gt_mask_tensor)
                total_iou += iou

                #epoch_loss += loss.item()
                print(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_train}  batch:{batch_idx}/{len(oneImage_dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')
                logging.info(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_train}  batch:{batch_idx}/{len(oneImage_dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')
                
        avg_val_loss = total_val_loss / total_num
        avg_accuracy = total_accuracy / total_num
        avg_iou = total_iou / total_num
        print(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')
        logging.info(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')

        # 验证阶段
        sam.eval()
        total_val_loss = 0
        total_accuracy = 0
        total_iou = 0
        total_num = 0
        with torch.no_grad():
            for image_index, (img_path, annotations) in tqdm(enumerate(val_image_dataloader, start=1), total=len(val_image_dataloader), desc=f'Val ==> Epoch:{epoch + 1}/{epochs}'):
                oneImage_dataset = COCODataset_OneImage(annotations)
                oneImage_dataloader = DataLoader(oneImage_dataset, batch_size=batch_size)
                total_num += len(oneImage_dataloader)
                for batch_idx, (rbox_tensor, gt_mask_tensor) in enumerate(oneImage_dataloader,start=1):  # point_tensor:b12    gt_mask_tensor:bhw
                    masks = process_single(sam, img_path[0], rbox_tensor).cpu()  # bchw
                    # 计算损失
                    focal = focal_loss(masks.squeeze(1), gt_mask_tensor)
                    dice = dice_loss(masks.squeeze(1), gt_mask_tensor)
                    loss = 20 * focal + dice
                    total_val_loss += loss.item()

                    accuracy = calculate_accuracy(masks.squeeze(1)> 0.0, gt_mask_tensor)
                    total_accuracy += accuracy

                    iou = calculate_iou(masks.squeeze(1)> 0.0, gt_mask_tensor)
                    total_iou += iou
                    print(f'VAL ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_val}  batch:{batch_idx}/{len(oneImage_dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')
                    logging.info(f'VAL ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_val}  batch:{batch_idx}/{len(oneImage_dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')


        avg_val_loss = total_val_loss / total_num
        avg_accuracy = total_accuracy / total_num
        avg_iou = total_iou / total_num
        print(f'VAL ==> Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')
        logging.info(f'VAL ==> Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')

        # 保存图像编码器的参数
        checkpoint_path = os.path.join(save_dir, f'sam_epoch_{epoch + 1 + 66}_lr4_un1000.pth')
        torch.save(sam.state_dict(), checkpoint_path)
        #print(f'Saved image encoder parameters to {checkpoint_path}')
        print(f'Saved sam parameters to {checkpoint_path}')
        logging.info(f'Saved sam parameters to {checkpoint_path}')


        if avg_iou>best_iou:
            best_epoch=epoch + 1
            best_loss=avg_val_loss
            best_accuracy=avg_accuracy
            best_iou=avg_iou
            # 保存图像编码器的参数
            #checkpoint_path = os.path.join(save_dir, f'image_encoder_best.pth')
            checkpoint_path = os.path.join(save_dir, f'sam_best_lr4_un1000.pth')
            #torch.save(sam.image_encoder.state_dict(), checkpoint_path)
            torch.save(sam.state_dict(), checkpoint_path)
            print(f'Saved sam parameters to {checkpoint_path}')
            logging.info(f'Saved sam parameters to {checkpoint_path}')

    print(f'训练已完成 Epoch：{epochs}, Batch: {batch_size}, num_train: {num_train}, num_val: {num_val}, best_epoch:{best_epoch}, best_loss:{best_loss}, best_accuracy:{best_accuracy}, best_iou:{best_iou} ')
    logging.info(f'训练已完成 Epoch：{epochs}, Batch: {batch_size}, num_train: {num_train}, num_val: {num_val}, best_epoch:{best_epoch}, best_loss:{best_loss}, best_accuracy:{best_accuracy}, best_iou:{best_iou} ')



# 获取当前脚本文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 拼接路径
#datasetPath = os.path.join(current_dir, "dataset", "data")
#save_Path= os.path.join(current_dir, "checkpoint_b")
#log_path = os.path.join(current_dir, "training_log_b.txt")

#datasetPath='C:\\Users\\yangxinyao\\Downloads\\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0'
#save_Path='C:\\Users\\yangxinyao\\Downloads\\checkpoint'
save_Path= os.path.join(current_dir, "checkpoint_new")
train(5,4,save_Path)

