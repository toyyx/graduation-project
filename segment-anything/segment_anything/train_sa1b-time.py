import logging
import math
import os
from typing import Tuple
import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
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


        #sam.prompt_encoder.rbox_point_embeddings[0].weight.data = sam.prompt_encoder.point_embeddings[2].weight.data.clone()
        #sam.prompt_encoder.rbox_point_embeddings[1].weight.data = sam.prompt_encoder.point_embeddings[3].weight.data.clone()
        #sam.prompt_encoder.rbox_point_embeddings[2].weight.data = sam.prompt_encoder.point_embeddings[3].weight.data.clone()
        #sam.prompt_encoder.rbox_point_embeddings[3].weight.data = sam.prompt_encoder.point_embeddings[2].weight.data.clone()
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
        #if 'rbox_point_embeddings' not in name and 'mask_decoder' not in name:
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
#def process_single(sam, image_file_path, rbox_tensor):
def process_single(sam, image_file_path, bbox_tensor, rbox_tensor):
    def xywh_to_xyxy(boxes):
        """
        将 xywh 格式的边界框转换为 xyxy 格式
        :param boxes: 形状为 (N, 4) 的张量，N 是边界框数量，每个边界框是 [x, y, w, h] 格式
        :return: 形状为 (N, 4) 的张量，每个边界框是 [x1, y1, x2, y2] 格式
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return torch.stack([x1, y1, x2, y2], dim=1)
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
    
    bbox_tensor =xywh_to_xyxy(bbox_tensor)
    bbox_tensor = torch.tensor(transform.apply_boxes(bbox_tensor.numpy(), original_size),device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配
    rbox_tensor = torch.tensor(transform.apply_rboxes(rbox_tensor.numpy(), original_size),device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        #boxes=None,
        boxes=bbox_tensor,
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

def process_single_image(sam, image_file_path):
    def xywh_to_xyxy(boxes):
        """
        将 xywh 格式的边界框转换为 xyxy 格式
        :param boxes: 形状为 (N, 4) 的张量，N 是边界框数量，每个边界框是 [x, y, w, h] 格式
        :return: 形状为 (N, 4) 的张量，每个边界框是 [x1, y1, x2, y2] 格式
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return torch.stack([x1, y1, x2, y2], dim=1)
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

    return image_embeddings,input_size,original_size

def process_single_mask(sam, image_embeddings,input_size,original_size, bbox_tensor, rbox_tensor):
    def xywh_to_xyxy(boxes):
        """
        将 xywh 格式的边界框转换为 xyxy 格式
        :param boxes: 形状为 (N, 4) 的张量，N 是边界框数量，每个边界框是 [x, y, w, h] 格式
        :return: 形状为 (N, 4) 的张量，每个边界框是 [x1, y1, x2, y2] 格式
        """
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        return torch.stack([x1, y1, x2, y2], dim=1)
    transform = ResizeLongestSide(sam.image_encoder.img_size)  # 创建一个 ResizeLongestSide 实例，用于将图像的最长边调整为模型期望的尺寸
    bbox_tensor =xywh_to_xyxy(bbox_tensor)
    bbox_tensor = torch.tensor(transform.apply_boxes(bbox_tensor.numpy(), original_size),device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配
    rbox_tensor = torch.tensor(transform.apply_rboxes(rbox_tensor.numpy(), original_size),device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        #boxes=None,
        boxes=bbox_tensor,
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

from dataset import SingleJSONDataset_with_rbox
def train(epochs,batch_size,save_Path):
    logging.basicConfig(filename='training_sa1b_combine_time.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    checkpoint_path=r'./checkpoint_new/sam_conbine_epoch_67_5lr4.pth'
    start_time = time.time()
    #checkpoint_path=None
    sam=init_model(checkpoint_path)
    init_time = time.time() - start_time
    print(f"初始化模型耗时: {init_time} 秒")
    logging.info(f"初始化模型耗时: {init_time} 秒")
    
    # 定义优化器，只优化可训练的参数
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sam.parameters()), lr=5e-4)

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    #dice_loss = DBSLoss()
    save_dir=save_Path
    best_iou=0
    best_accuracy=0
    best_loss = 1000
    best_epoch=0

    dataset_folder_path=r'/root/autodl-tmp/SA-1B/example'
    all_image_files = [f for f in os.listdir(dataset_folder_path) if f.endswith('.jpg')]
    num_train = 1000
    num_val = 100
   

    # 划分数据集
    train_image_files = all_image_files[:num_train]
    val_image_files = all_image_files[-num_val:]
    
    print('start reading json......')
    start_time = time.time()
    # 存储所有 JSON 文件的数据
    train_jsondata = []
    # 一次性读取所有 JSON 文件
    for image_filename in train_image_files:
        json_filename = os.path.splitext(image_filename)[0] + '.json'
        json_file_path = os.path.join(dataset_folder_path, json_filename)
        # 检查对应的JSON文件是否存在
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                    train_jsondata.append(data)
            except json.JSONDecodeError:
                print(f"JSON 解析错误: {json_file_path}")
            except Exception as e:
                print(f"读取文件 {json_file_path} 时出错: {e}")
                
    # 存储所有 JSON 文件的数据
    val_jsondata = []
    # 一次性读取所有 JSON 文件
    for image_filename in val_image_files:
        json_filename = os.path.splitext(image_filename)[0] + '.json'
        json_file_path = os.path.join(dataset_folder_path, json_filename)
        # 检查对应的JSON文件是否存在
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                    val_jsondata.append(data)
            except json.JSONDecodeError:
                print(f"JSON 解析错误: {json_file_path}")
            except Exception as e:
                print(f"读取文件 {json_file_path} 时出错: {e}")
                
    read_json_time = time.time() - start_time
    print(f"读取 JSON 文件耗时: {read_json_time} 秒")
    logging.info(f"读取 JSON 文件耗时: {read_json_time} 秒")
    print('finish reading json')
    
    for epoch in range(epochs):
        # 训练阶段
        sam.train()
        total_train_loss = 0
        total_accuracy = 0
        total_iou = 0
        total_num = 0
        
        """
         # 遍历文件夹中的所有文件
        for image_index, image_filename in tqdm(enumerate(train_image_files, start=1), total=len(train_image_files), desc=f'Training ==> Epoch:{epoch + 1}/{epochs}'):
            json_filename = os.path.splitext(image_filename)[0] + '.json'
            json_file_path = os.path.join(dataset_folder_path, json_filename)
            image_file_path = os.path.join(dataset_folder_path, image_filename)

            # 检查对应的JSON文件是否存在
            if os.path.exists(json_file_path):
                dataset = SingleJSONDataset_with_rbox(json_file_path)
        """
        
            
        # 遍历所有图像文件
        for image_index, (image_filename, data) in tqdm(enumerate(zip(train_image_files, train_jsondata), start=1),total=len(train_image_files),desc=f'Training ==> Epoch:{epoch + 1}/{epochs}'):
            start_time = time.time()
            dataset = SingleJSONDataset_with_rbox(data)
            dataloader = DataLoader(dataset, batch_size=batch_size,pin_memory=True)
            total_num += len(dataloader)
            image_file_path = os.path.join(dataset_folder_path, image_filename)
            image_embeddings=None
            image_prepare_time = time.time() - start_time
            print(f"准备图片dataset 文件耗时: {image_prepare_time} 秒")
            logging.info(f"准备图片dataset 文件耗时: {image_prepare_time} 秒")
            #for batch_idx, (rbox_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):#point_tensor:b12    gt_mask_tensor:bhw
            for batch_idx, (bbox_tensor, rbox_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):#point_tensor:b12    gt_mask_tensor:bhw
                batch_start_time=time.time()
                if image_embeddings is None:
                    start_time = time.time()
                    with torch.no_grad():
                        image_embeddings,input_size,original_size=process_single_image(sam, image_file_path)
                    image_embeddings_time = time.time() - start_time
                    print(f"图片嵌入耗时: {image_embeddings_time} 秒")
                    logging.info(f"图片嵌入耗时: {image_embeddings_time} 秒")
                start_time = time.time()
                masks=process_single_mask(sam, image_embeddings,input_size,original_size, bbox_tensor, rbox_tensor).cpu()
                mask_time = time.time() - start_time
                print(f"mask耗时: {mask_time} 秒")
                logging.info(f"mask耗时: {mask_time} 秒")
                #masks = process_single(sam, image_file_path, rbox_tensor).cpu()#bchw
                #masks = process_single(sam, image_file_path, bbox_tensor, rbox_tensor).cpu()#bchw
                #masks = masks > 0.0  # 根据掩码阈值将掩码转换为二进制掩码
                
                loss_start_time = time.time()
                
                # 计算损失
                focal = focal_loss(masks.squeeze(1) , gt_mask_tensor)
                dice = dice_loss(masks.squeeze(1) , gt_mask_tensor)
                loss = 20 * focal + dice
                total_train_loss += loss.item()
                
                compute_loss_time = time.time() - loss_start_time
                print(f"计算loss耗时: {compute_loss_time} 秒")
                logging.info(f"计算loss耗时: {compute_loss_time} 秒")
                
                back_start_time = time.time()
                # 反向传播和优化
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                back_time = time.time() - back_start_time
                print(f"反向耗时: {back_time} 秒")
                logging.info(f"反向耗时: {back_time} 秒")

                other_start_time = time.time()
                accuracy = calculate_accuracy(masks.squeeze(1)> 0.0, gt_mask_tensor)
                total_accuracy += accuracy
                iou = calculate_iou(masks.squeeze(1)> 0.0, gt_mask_tensor)
                total_iou += iou
                other_time = time.time() - other_start_time
                print(f"其余耗时: {other_time} 秒")
                logging.info(f"其余耗时: {other_time} 秒")
                
                
                loss_time = time.time() - loss_start_time
                print(f"loss耗时: {loss_time} 秒")
                logging.info(f"loss耗时: {loss_time} 秒")
                print(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_train}  batch:{batch_idx}/{len(dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')
                logging.info(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_train}  batch:{batch_idx}/{len(dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')
                
                batch_time = time.time() - batch_start_time
                print(f"batch耗时: {batch_time} 秒")
                logging.info(f"batch耗时: {batch_time} 秒")
        avg_train_loss = total_train_loss / total_num
        avg_accuracy = total_accuracy / total_num
        avg_iou = total_iou / total_num
        print(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')
        logging.info(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')

        # 验证阶段
        sam.eval()
        total_val_loss = 0
        total_accuracy = 0
        total_iou = 0
        total_num = 0
        with torch.no_grad():
            # 遍历所有图像文件
            """
             # 遍历文件夹中的所有文件
            for image_index, image_filename in tqdm(enumerate(val_image_files, start=1), total=len(val_image_files), desc=f'Val ==> Epoch:{epoch + 1}/{epochs}'):
                json_filename = os.path.splitext(image_filename)[0] + '.json'
                json_file_path = os.path.join(dataset_folder_path, json_filename)
                image_file_path = os.path.join(dataset_folder_path, image_filename)

                # 检查对应的JSON文件是否存在
                if os.path.exists(json_file_path):
                    dataset = SingleJSONDataset_with_rbox(json_file_path)
                    
            """
            for image_index, (image_filename, data) in tqdm(enumerate(zip(val_image_files, val_jsondata), start=1),total=len(val_image_files),desc=f'Val ==> Epoch:{epoch + 1}/{epochs}'):
                dataset = SingleJSONDataset_with_rbox(data)
                dataloader = DataLoader(dataset, batch_size=batch_size,pin_memory=True)
                total_num += len(dataloader)
                image_file_path = os.path.join(dataset_folder_path, image_filename)
                #for batch_idx, (rbox_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):#point_tensor:b12    gt_mask_tensor:bhw
                for batch_idx, (bbox_tensor, rbox_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):#point_tensor:b12    gt_mask_tensor:bhw
                    #masks = process_single(sam, image_file_path, rbox_tensor).cpu()  # bchw
                    masks = process_single(sam, image_file_path, bbox_tensor, rbox_tensor).cpu()#bchw
                    # 计算损失
                    focal = focal_loss(masks.squeeze(1), gt_mask_tensor)
                    dice = dice_loss(masks.squeeze(1), gt_mask_tensor)
                    loss = 20 * focal + dice
                    total_val_loss += loss.item()

                    accuracy = calculate_accuracy(masks.squeeze(1)> 0.0, gt_mask_tensor)
                    total_accuracy += accuracy

                    iou = calculate_iou(masks.squeeze(1)> 0.0, gt_mask_tensor)
                    total_iou += iou
                    print(f'VAL ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')
                    logging.info(f'VAL ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')


        avg_val_loss = total_val_loss / total_num
        avg_accuracy = total_accuracy / total_num
        avg_iou = total_iou / total_num
        print(f'VAL ==> Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')
        logging.info(f'VAL ==> Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')

        # 保存图像编码器的参数
        checkpoint_path = os.path.join(save_dir, f'sam_conbine_epoch_{epoch + 1 + 67}_5lr4.pth')
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
            checkpoint_path = os.path.join(save_dir, f'sam_best_conbine_epoch_5lr4.pth')
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
train(2,16,save_Path)

