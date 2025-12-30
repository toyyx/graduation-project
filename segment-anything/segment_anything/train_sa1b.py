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
import json
from build_sam import sam_model_registry
from dataset import SingleJSONDataset
from utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import gaussian_blur
import numpy as np

def init_logger(filename):
    # 创建第一个日志记录器
    logger1 = logging.getLogger(filename)
    logger1.setLevel(logging.INFO)

    # 创建第一个日志记录器的文件处理器
    file_handler1 = logging.FileHandler(filename)
    file_handler1.setLevel(logging.INFO)

    # 创建第一个日志记录器的格式化器
    formatter1 = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler1.setFormatter(formatter1)

    # 将处理器添加到第一个日志记录器
    logger1.addHandler(file_handler1)

    return logger1


def sort_rbox_points(rbox):
    """
    按 top_point、right_point、bottom_point、left_point 的顺序排序 rbox 的点
    :param rbox: 旋转包围框的四个点坐标
    :return: 排序后的旋转包围框的四个点坐标
    """
    rbox = np.array(rbox)
    x_values = rbox[:, 0]
    y_values = rbox[:, 1]
    # 判断 x 或 y 值是否都不相同
    if len(set(x_values)) == 4 and len(set(y_values)) == 4:
        top_point = rbox[np.argmin(rbox[:, 1])]
        right_point = rbox[np.argmax(rbox[:, 0])]
        bottom_point = rbox[np.argmax(rbox[:, 1])]
        left_point = rbox[np.argmin(rbox[:, 0])]
        return [top_point.tolist(), right_point.tolist(), bottom_point.tolist(), left_point.tolist()]
    else:
        # 手动设定矩形
        min_x = np.min(x_values)
        max_x = np.max(x_values)
        min_y = np.min(y_values)
        max_y = np.max(y_values)

        top_point = [min_x, min_y]
        right_point = [max_x, min_y]
        bottom_point = [max_x, max_y]
        left_point = [min_x, max_y]
        return [top_point, right_point, bottom_point, left_point]


def get_rbox_from_binary_mask(binary_mask, img_width, img_height):
    # 将 binary_mask 从 Tensor 转换为 numpy 数组
    binary_mask = binary_mask.cpu().numpy()
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return [[0, 0], [0, 0], [0, 0], [0, 0]]
    all_points = np.vstack(contours)
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    # 裁剪超出图片范围的点
    box[:, 0] = np.clip(box[:, 0], 0, img_width)
    box[:, 1] = np.clip(box[:, 1], 0, img_height)
    # 排序 rbox 的点
    sorted_box = sort_rbox_points(box)
    return sorted_box

def get_angle_from_rbox(rbox):
    """
    分批次计算多个 rbox 的方向角度（0 - 180 度）
    :param rboxes: 形状为 (4, 2) 的 numpy 数组，n 是 rbox 的数量
    :return: 形状为 (1,) 的 numpy 数组，包含每个 rbox 的方向角度
    """
    rbox =  np.array(rbox)
    # 计算协方差矩阵
    cov_matrix = np.cov(rbox.T)
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # 找到最大特征值对应的特征向量，它代表了主要方向
    max_eigenvalue_index = np.argmax(eigenvalues)
    principal_vector = eigenvectors[:, max_eigenvalue_index]
    # 计算角度（弧度）
    angle = np.arctan2(principal_vector[1], principal_vector[0])
    # 将角度转换到 0 - 180 度范围
    angle = np.degrees(angle)
    if angle < 0:
        angle += 180
    return angle

def calculate_angle_from_masks(masks, img_width, img_height):#bhw
    batch_size = masks.size(0)
    angles=[]
    for i in range(batch_size):
        binary_mask=masks[i]
        rbox=get_rbox_from_binary_mask(binary_mask, img_width, img_height)
        angle=get_angle_from_rbox(rbox)
        angles.append(angle)
    return angles

def calculate_angle_from_rboxes_tensor(rboxes_tensor):#bhw
    batch_size = rboxes_tensor.size(0)
    angles = []
    for i in range(batch_size):
        rbox = rboxes_tensor[i].tolist()
        angle = get_angle_from_rbox(rbox)
        angles.append(angle)
    return angles


def angular_mae_loss(pred_angles, gt_angles):#0.01
    """
    计算考虑角度周期性的平均绝对误差损失
    :param pred_angles: 预测角度的张量
    :param gt_angles: 真实角度的张量
    :return: 角度平均绝对误差损失
    """
    pred_angles = torch.tensor(pred_angles)
    gt_angles = torch.tensor(gt_angles)
    angle_diffs = torch.abs(pred_angles - gt_angles)
    # 处理角度周期性
    angle_diffs = torch.min(angle_diffs, 360 - angle_diffs)
    return torch.mean(angle_diffs)


def angular_mse_loss(pred_angles, gt_angles):#0.00001
    """
    计算考虑角度周期性的均方误差损失
    :param pred_angles: 预测角度的张量
    :param gt_angles: 真实角度的张量
    :return: 角度均方误差损失
    """
    pred_angles = torch.tensor(pred_angles)
    gt_angles = torch.tensor(gt_angles)
    angle_diffs = torch.abs(pred_angles - gt_angles)
    # 处理角度周期性
    angle_diffs = torch.min(angle_diffs, 360 - angle_diffs)
    return torch.mean(angle_diffs ** 2)

def angular_cosine_loss(pred_angles, gt_angles):
    """
    计算角度的余弦相似度损失
    :param pred_angles: 预测角度的张量
    :param gt_angles: 真实角度的张量
    :return: 角度余弦相似度损失
    """
    pred_angles = torch.tensor(pred_angles)
    gt_angles = torch.tensor(gt_angles)

    # 手动实现角度到弧度的转换
    pred_radians = pred_angles * (math.pi / 180)
    gt_radians = gt_angles * (math.pi / 180)
    pred_vectors = torch.stack([torch.cos(pred_radians), torch.sin(pred_radians)], dim=-1)
    gt_vectors = torch.stack([torch.cos(gt_radians), torch.sin(gt_radians)], dim=-1)
    #pred_vectors = torch.stack([torch.cos(torch.radians(pred_angles)), torch.sin(torch.radians(pred_angles))], dim=-1)
    #gt_vectors = torch.stack([torch.cos(torch.radians(gt_angles)), torch.sin(torch.radians(gt_angles))], dim=-1)
    cos_sim = torch.nn.functional.cosine_similarity(pred_vectors, gt_vectors, dim=-1)
    return 1 - torch.mean(cos_sim)


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
    def __init__(self, sigma=4, smooth=1e-5):
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
        weight_map = weight_map / (max_values + 1e-6)
         # 去除通道维度，得到 (N, H, W) 形状的权重图
        weight_map = weight_map.squeeze(1)  # (N, H, W)
        
        # 只保留掩码内部的权重，外部置为 1
        #weight_map = weight_map * mask.squeeze(1) + 1.0
        weight_map = (weight_map * mask.squeeze(1)) / 5 + 1.0

        return weight_map


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

    #sam.prompt_encoder.bbox_point_embeddings_origin[0].weight.data = sam.prompt_encoder.point_embeddings[2].weight.data.clone()
    #sam.prompt_encoder.bbox_point_embeddings_origin[1].weight.data = sam.prompt_encoder.point_embeddings[3].weight.data.clone()
    
    #sam.prompt_encoder.rbox_mask_embed.weight.data = sam.prompt_encoder.no_mask_embed.weight.data.clone()
    
    
    for name, param in sam.named_parameters():
        if 'rbox_point_embeddings' in name and 'rbox_point_embeddings_origin' not in name:
            param.requires_grad = True
            print(f"可训练参数名: {name}")
        else:
            param.requires_grad = False
    
    """
    # 冻结原有参数no_mask_embed
    for name, param in sam.named_parameters():
        if 'bbox_point_embeddings_origin' in name:
        #if 'bbox_point_embeddings_origin' in name or ('rbox_point_embeddings' in name and 'rbox_point_embeddings_origin' not in name):
            param.requires_grad = True
            print(f"可训练参数名: {name}")
        else:
            param.requires_grad = False
    """
    """
    if 'rbox_point_embeddings' not in name or 'rbox_point_embeddings_origin' in name:
    #if 'rbox_point_embeddings_origin' not in name and 'rbox_mask_embed' not in name:
    #if 'rbox_mask_embed' not in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
        print(f"可训练参数名: {name}")
    """
        
            

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
    
    #bbox_tensor =xywh_to_xyxy(bbox_tensor)
    bbox_tensor = torch.tensor(transform.apply_boxes(bbox_tensor.numpy(), original_size),device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配
    rbox_tensor = torch.tensor(transform.apply_rboxes(rbox_tensor.numpy(), original_size),device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        #boxes=None,
        boxes=bbox_tensor,
        rboxes=None,
        rboxes_origin=rbox_tensor,
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

    return image_embeddings.detach(),input_size,original_size

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
        boxes=bbox_tensor,
        #boxes=bbox_tensor,
        boxes_origin=None,
        rboxes=rbox_tensor,
        rboxes_origin=None,
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
    #logging.basicConfig(filename='training_sa1b_combine_DBSL.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    training_logger = init_logger("training_sa1b_DBSL_sigma4_div5_rbox.log")
    train_avg_logger = init_logger("train_avg_sa1b_DBSL_sigma4_div5_rbox.log")
    val_avg_logger = init_logger("val_avg_sa1b_DBSL_sigma4_div5_rbox.log")
    checkpoint_path=r'/root/autodl-tmp/sam_checkpoint/sam_best_conbine_epoch_lr5_DBSL_sigma4_div5_rbox.pth'
    #checkpoint_path=None
    sam=init_model(checkpoint_path)
    # 定义优化器，只优化可训练的参数
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, sam.parameters()), lr=1e-7)

    focal_loss = FocalLoss()
    #dice_loss = DiceLoss()
    dice_loss = DBSLoss()
    save_dir=save_Path
    best_iou=0
    best_accuracy=0
    best_loss = 1000
    best_epoch=0
    
    train_dataset_folder_path = r'/root/autodl-tmp/SA-1B/train-100'
    val_dataset_folder_path = r'/root/autodl-tmp/SA-1B/val-un10'
    train_image_files = [f for f in os.listdir(train_dataset_folder_path) if f.endswith('.jpg')]
    val_image_files = [f for f in os.listdir(val_dataset_folder_path) if f.endswith('.jpg')]
    num_train = 100
    num_val = 10
    # 划分数据集
    train_image_files = train_image_files[:num_train]
    val_image_files = val_image_files[:num_val]

    #dataset_folder_path=r'/root/autodl-tmp/SA-1B/example'
    #all_image_files = [f for f in os.listdir(dataset_folder_path) if f.endswith('.jpg')]
    #num_train = 100
    #num_val = 10
   

    # 划分数据集
    #train_image_files = all_image_files[:num_train]
    #val_image_files = all_image_files[-num_val:]
    
    print('start reading json......')
    # 存储所有 JSON 文件的数据
    train_jsondata = []
    # 一次性读取所有 JSON 文件
    for image_filename in train_image_files:
        json_filename = os.path.splitext(image_filename)[0] + '.json'
        json_file_path = os.path.join(train_dataset_folder_path, json_filename)
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
        json_file_path = os.path.join(val_dataset_folder_path, json_filename)
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
            dataset = SingleJSONDataset_with_rbox(data)
            dataloader = DataLoader(dataset, batch_size=batch_size,pin_memory=True)
            total_num += len(dataloader)
            image_file_path = os.path.join(train_dataset_folder_path, image_filename)
            image_embeddings=None
            #for batch_idx, (rbox_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):#point_tensor:b12    gt_mask_tensor:bhw
            for batch_idx, (bbox_tensor, rbox_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):#point_tensor:b12    gt_mask_tensor:bhw
                if image_embeddings is None:
                    image_embeddings,input_size,original_size=process_single_image(sam, image_file_path)
                masks=process_single_mask(sam, image_embeddings,input_size,original_size, bbox_tensor, rbox_tensor).cpu()
                
                #masks = process_single(sam, image_file_path, rbox_tensor).cpu()#bchw
                #masks = process_single(sam, image_file_path, bbox_tensor, rbox_tensor).cpu()#bchw
                #masks = masks > 0.0  # 根据掩码阈值将掩码转换为二进制掩码
                
                #gt_angles=calculate_angle_from_rboxes_tensor(rbox_tensor)
                #pred_angles=calculate_angle_from_masks(masks.squeeze(1)> 0.0,original_size[1],original_size[0])
                #angle_loss=angular_mse_loss(pred_angles,gt_angles)
                # 计算损失
                focal = focal_loss(masks.squeeze(1) , gt_mask_tensor)
                dice = dice_loss(masks.squeeze(1) , gt_mask_tensor)
                loss = 20 * focal + dice
                total_train_loss += loss.item()

                # 反向传播和优化
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                """
                # 计算梯度范数
                total_norm = 0
                for p in sam.parameters():
                    if p.grad is not None and p.requires_grad:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f'Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_train}  batch:{batch_idx}/{len(dataloader)}, Gradient norm = {total_norm}')
                """
                optimizer.step()

                accuracy = calculate_accuracy(masks.squeeze(1)> 0.0, gt_mask_tensor)
                total_accuracy += accuracy
                iou = calculate_iou(masks.squeeze(1)> 0.0, gt_mask_tensor)
                total_iou += iou

                print(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_train}  batch:{batch_idx}/{len(dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')
                training_logger.info(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_train}  batch:{batch_idx}/{len(dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')
                
        avg_train_loss = total_train_loss / total_num
        avg_accuracy = total_accuracy / total_num
        avg_iou = total_iou / total_num
        print(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')
        train_avg_logger.info(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')

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
                image_file_path = os.path.join(val_dataset_folder_path, image_filename)
                image_embeddings=None
                #for batch_idx, (rbox_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):#point_tensor:b12    gt_mask_tensor:bhw
                for batch_idx, (bbox_tensor, rbox_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):#point_tensor:b12    gt_mask_tensor:bhw
                    if image_embeddings is None:
                        image_embeddings,input_size,original_size=process_single_image(sam, image_file_path)
                    masks=process_single_mask(sam, image_embeddings,input_size,original_size, bbox_tensor, rbox_tensor).cpu()
                    #masks = process_single(sam, image_file_path, rbox_tensor).cpu()  # bchw
                    #masks = process_single(sam, image_file_path, bbox_tensor, rbox_tensor).cpu()#bchw
                    #gt_angles=calculate_angle_from_rboxes_tensor(rbox_tensor)
                    #pred_angles=calculate_angle_from_masks(masks.squeeze(1)> 0.0,original_size[1],original_size[0])
                    #angle_loss=angular_mse_loss(pred_angles,gt_angles)
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
                    training_logger.info(f'VAL ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')


        avg_val_loss = total_val_loss / total_num
        avg_accuracy = total_accuracy / total_num
        avg_iou = total_iou / total_num
        print(f'VAL ==> Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')
        val_avg_logger.info(f'VAL ==> Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')

        # 保存图像编码器的参数
        #checkpoint_path = os.path.join(save_dir, f'sam_conbine_epoch_{epoch + 1}_lr5_DBSL_sigma4_div3.pth')
        #torch.save(sam.state_dict(), checkpoint_path)
        #print(f'Saved image encoder parameters to {checkpoint_path}')
        #print(f'Saved sam parameters to {checkpoint_path}')
        #training_logger.info(f'Saved sam parameters to {checkpoint_path}')


        if avg_iou>best_iou:
            best_epoch=epoch + 1
            best_loss=avg_val_loss
            best_accuracy=avg_accuracy
            best_iou=avg_iou
            # 保存图像编码器的参数
            checkpoint_path = os.path.join(save_dir, f'sam_best_conbine_epoch_lr7_DBSL_sigma4_div5_rbox.pth')
            torch.save(sam.state_dict(), checkpoint_path)
            print(f'Saved sam parameters to {checkpoint_path}')
            training_logger.info(f'Saved sam parameters to {checkpoint_path}')

    print(f'训练已完成 Epoch：{epochs}, Batch: {batch_size}, num_train: {num_train}, num_val: {num_val}, best_epoch:{best_epoch}, best_loss:{best_loss}, best_accuracy:{best_accuracy}, best_iou:{best_iou} ')
    val_avg_logger.info(f'训练已完成 Epoch：{epochs}, Batch: {batch_size}, num_train: {num_train}, num_val: {num_val}, best_epoch:{best_epoch}, best_loss:{best_loss}, best_accuracy:{best_accuracy}, best_iou:{best_iou} ')



# 获取当前脚本文件所在的目录
#current_dir = os.path.dirname(os.path.abspath(__file__))

# 拼接路径
#datasetPath = os.path.join(current_dir, "dataset", "data")
#save_Path= os.path.join(current_dir, "checkpoint_b")
#log_path = os.path.join(current_dir, "training_log_b.txt")

#datasetPath='C:\\Users\\yangxinyao\\Downloads\\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0'
#save_Path='C:\\Users\\yangxinyao\\Downloads\\checkpoint'

#save_Path = os.path.join(current_dir, "checkpoint_new")

save_Path = r'/root/autodl-tmp/sam_checkpoint'
train(10,16,save_Path)

