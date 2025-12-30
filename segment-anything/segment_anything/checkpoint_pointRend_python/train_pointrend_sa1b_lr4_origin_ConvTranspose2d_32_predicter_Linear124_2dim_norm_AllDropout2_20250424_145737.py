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
from typing import Any, Optional, Tuple, Type
#torch.backends.cudnn.enabled = True
#torch.backends.cudnn.benchmark = True
import os
import shutil
from datetime import datetime

def save_backup_file(file_name, append_str):
    """
    保存文件的备份，在原文件名后添加指定后缀和时间信息，并复制到指定目录。

    :param file_name: 要备份的文件名
    :param append_str: 要添加到文件名的后缀字符串
    :return: 备份文件的完整路径，如果操作失败则返回 None
    """
    backup_dir = "checkpoint_pointRend_python"  # 固定的备份目录
    try:
        # 检查文件是否存在
        if not os.path.exists(file_name):
            print(f"文件 {file_name} 不存在，无法进行备份。")
            return None

        # 创建备份目录（如果不存在）
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)

        # 拆分文件名和扩展名
        base_name, ext = os.path.splitext(os.path.basename(file_name))

        # 获取当前时间
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 生成备份文件名
        backup_name = f"{base_name}_{append_str}_{timestamp}{ext}"

        # 生成备份文件的完整路径
        backup_path = os.path.join(backup_dir, backup_name)

        # 复制文件
        shutil.copy2(file_name, backup_path)

        print(f"已成功保存备份文件：{backup_path}")
        return backup_path
    except Exception as e:
        print(f"保存备份文件时出现错误：{e}")
        return None

class PositionEmbeddingRandom():
    def __init__(self, num_pos_feats: int = 32, scale: Optional[float] = 1.0) -> None:
        self.positional_encoding_gaussian_matrix= scale * torch.randn((2, num_pos_feats//2),device='cuda' if torch.cuda.is_available() else 'cpu')# 注册一个缓冲区，用于存储随机生成的高斯矩阵
                  
    def _pe_encoding(self,coords: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # coords (torch.Tensor): 归一化后的坐标张量，形状为 d_1 x ... x d_n x 2
            """Positionally encode points that are normalized to [0,1]."""  # 对归一化到 [0, 1] 范围内的点进行位置编码
            # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
            coords = 2 * coords - 1  # 假设输入的坐标在 [0, 1]^2 正方形内，将坐标从 [0, 1] 范围转换到 [-1, 1] 范围
            coords = coords @ self.positional_encoding_gaussian_matrix  # 将坐标与高斯矩阵相乘，得到新的坐标表示
            coords = 2 * np.pi * coords  # 将坐标乘以 2π，用于后续的正弦和余弦计算
            # outputs d_1 x ... x d_n x C shape
            # 对坐标分别取正弦和余弦值，并在最后一个维度上拼接
            # 最终输出形状为 d_1 x ... x d_n x C 的位置编码张量
            return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)  # torch.Tensor: 位置编码后的张量，形状为 d_1 x ... x d_n x C
    
    def compute_pe_for_sampled_points(self,coords):
        """
        为给定的采样点坐标计算位置编码
        :param coords: 采样点坐标，形状为 (b, n, 2)，格式为 (h, w)
        :param h: 图像的高度
        :param w: 图像的宽度
        :return: 位置编码结果，形状为 (b, n, 2 * num_pos_feats)
        """
        with torch.no_grad():
            if isinstance(coords, torch.Tensor):
                # 转换坐标顺序 (h, w) 到 (w, h)
                normalized_coords = torch.stack([coords[..., 1], coords[..., 0]], dim=-1)  # bn2

                return self._pe_encoding(normalized_coords)  # bnd    #nd
            elif isinstance(coords, list):
                result = []
                for coord in coords:
                    normalized_coord = torch.stack([coord[..., 1], coord[..., 0]], dim=-1)  # n2
                    result.append(self._pe_encoding(normalized_coord))
                return result

# torch.distributed.init_process_group(backend="gloo")
# 设置环境变量，控制多线程计算的线程数
# os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
# os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

def update_masks(pred_masks, result):
    B, C, H, W = pred_masks.shape
    points_idx = result["idx"]
    rend = result["rend"]

    if isinstance(points_idx, torch.Tensor) and isinstance(rend, torch.Tensor):
        # 处理 points_idx 为 bn 和 rend 为 b1n 张量的情况
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)
        pred_masks = (pred_masks.reshape(B, C, -1)
                      .scatter_(2, points_idx, rend)
                      .view(B, C, H, W))
        return pred_masks
    elif isinstance(points_idx, list) and isinstance(rend, list):
        # 处理 points_idx 为列表和 rend 为列表的情况
        new_pred_masks = []
        for i in range(B):
            current_points_idx = points_idx[i].unsqueeze(0).expand(C, -1)  # c，<=n
            current_rend = rend[i]  # 1,<=n
            current_pred_mask = (pred_masks[i].reshape(C, -1)  # C,H*W
                                 .scatter_(1, current_points_idx, current_rend)
                                 .view(C, H, W))
            new_pred_masks.append(current_pred_mask)
        new_pred_masks = torch.stack(new_pred_masks, dim=0)
        return new_pred_masks
    else:
        raise ValueError("points_idx 和 rend 必须同时为张量或同时为列表")


def calculate_binary_cross_entropy_with_logits(result_rend, gt_points, reduction='mean'):
    if isinstance(result_rend, torch.Tensor) and isinstance(gt_points, torch.Tensor):
        # 处理 result_rend 和 gt_points 为张量的情况
        loss = F.binary_cross_entropy_with_logits(result_rend, (gt_points > 0.0).float(), reduction=reduction)
        return loss
    elif isinstance(result_rend, list) and isinstance(gt_points, list):
        # 处理 result_rend 和 gt_points 为列表的情况
        total_loss = 0
        total_elements = 0
        for rend, gt in zip(result_rend, gt_points):
            rend = rend.unsqueeze(0)
            gt = gt.unsqueeze(0)
            # 这里强制使用 'sum' 来累加每个元素的损失
            single_loss = F.binary_cross_entropy_with_logits(rend, (gt > 0.0).float(), reduction='sum')
            total_loss += single_loss
            # 累加每个元素最后一维的大小
            total_elements += rend.shape[1]
        if reduction == 'mean':
            return total_loss / total_elements
        elif reduction == 'sum':
            return total_loss
        else:
            raise ValueError("当输入为列表时，reduction 仅支持 'mean' 或 'sum'")
    else:
        raise ValueError("result_rend 和 gt_points 必须同时为张量或列表")


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


def calculate_accuracy(pred_mask, gt_mask):
    pred_mask = pred_mask.bool()
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
        if isinstance(inputs, torch.Tensor): 
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
        else:
            batch_losses = []
            for input_batch, target_batch in zip(inputs, targets):
                input_batch = input_batch.unsqueeze(0)
                target_batch = (target_batch.unsqueeze(0) > 0.0).float()
                BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(input_batch, target_batch)
                pt = torch.exp(-BCE_loss)
                F_loss_batch = self.alpha * (1 - pt) ** self.gamma * BCE_loss
                batch_losses.append(F_loss_batch.mean())
            return sum(batch_losses) / len(batch_losses)

# 定义 Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        smooth = 1e-5
        if isinstance(inputs, torch.Tensor): 
            inputs = torch.sigmoid(inputs)  # 将输入转换为概率值
            #inputs = (inputs> 0.0).float()  # 将输入转换为概率值
            # 对每个样本单独计算交集
            intersection = (inputs * targets).sum(dim=(1, 2))
            # 对每个样本单独计算输入和目标的和
            inputs_sum = inputs.sum(dim=(1, 2))
            targets_sum = targets.sum(dim=(1, 2))
            dice = (2. * intersection + smooth) / (inputs_sum + targets_sum + smooth)
            return 1 - dice.mean()
        else:
            result=[]
            for rend, gt in zip(inputs, targets):
                rend = rend.unsqueeze(0)
                gt = (gt.unsqueeze(0) > 0.0).float()
                rend = torch.sigmoid(rend)  # 将输入转换为概率值
                 # 对每个样本单独计算交集
                intersection = (rend * gt).sum(dim=(1, 2))
                # 对每个样本单独计算输入和目标的和
                inputs_sum = rend.sum(dim=(1, 2))
                targets_sum = gt.sum(dim=(1, 2))
                dice = (2. * intersection + smooth) / (inputs_sum + targets_sum + smooth)
                result.append(dice.mean())
            return 1-sum(result) / len(result)

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

    # 加载SAM模型
    sam = sam_model_registry[model_type]()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device)

    if checkpoint_path is None:
        checkpoint = torch.load(sam_checkpoint_path, map_location=device)

        log = sam.load_state_dict(checkpoint, strict=False)

        print("Model loaded from {} \n => {}".format(sam_checkpoint_path, log))
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        log = sam.load_state_dict(checkpoint, strict=False)

        print("Model loaded from {} \n => {}".format(checkpoint_path, log))

    # 冻结原有参数no_mask_embed
    for name, param in sam.named_parameters():
        param.requires_grad = False

    #total_trainable_params = 0
    #for name, param in sam.named_parameters():
        # param.requires_grad = True
        #print(f"可训练参数名: {name} {param.numel()}")
        #total_trainable_params += param.numel()

    #print(f"可训练参数量: {total_trainable_params}")

    return sam


from pointrend_test import PointRend, PointHead, point_sample, StandardPointHead, PointHead_try, PointRend_try, point_sample_by_idx


def init_pointrend(checkpoint_path=None):
    pointRend = PointRend_try(head=StandardPointHead(), training=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pointRend.to(device)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        log = pointRend.load_state_dict(checkpoint, strict=False)

        print("Model loaded from {} \n => {}".format(checkpoint_path, log))

    total_trainable_params = 0
    for name, param in pointRend.named_parameters():
        param.requires_grad = True
        print(f"可训练参数名: {name} {param.numel()}")
        total_trainable_params += param.numel()

    print(f"可训练参数量: {total_trainable_params}")

    return pointRend


import cv2


# def process_single(sam, image_file_path, rbox_tensor):
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
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :,
                        :]  # 调整张量的维度顺序为 BCHW（批次、通道、高度、宽度），并确保内存连续

    original_size = image.shape[:2]  # 保存原始图像的尺寸
    input_size = tuple(input_image_torch.shape[-2:])  # 保存输入图像（变换后）的尺寸
    input_image = sam.preprocess(input_image_torch)  # 对输入图像进行预处理，如归一化、填充等操作
    image_embeddings = sam.image_encoder(input_image)

    # bbox_tensor =xywh_to_xyxy(bbox_tensor)
    bbox_tensor = torch.tensor(transform.apply_boxes(bbox_tensor.numpy(), original_size),
                               device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配
    rbox_tensor = torch.tensor(transform.apply_rboxes(rbox_tensor.numpy(), original_size),
                               device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        # boxes=None,
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

    input_image, target_size = transform.apply_image_return_target_size(
        image)  # 应用 ResizeLongestSide 变换，将图像的最长边调整为模型期望的尺寸
    input_image_torch = torch.as_tensor(input_image, device=sam.device)  # 将 NumPy 数组转换为 PyTorch 张量，并将其放置在与模型相同的设备上
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :,
                        :]  # 调整张量的维度顺序为 BCHW（批次、通道、高度、宽度），并确保内存连续

    original_size = image.shape[:2]  # 保存原始图像的尺寸
    input_size = tuple(input_image_torch.shape[-2:])  # 保存输入图像（变换后）的尺寸
    input_image = sam.preprocess(input_image_torch)  # 对输入图像进行预处理，如归一化、填充等操作
    image_embeddings = sam.image_encoder(input_image)

    return image_embeddings, input_size, original_size, target_size


def process_single_mask(sam, image_embeddings, input_size, original_size, bbox_tensor, rbox_tensor):
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
    bbox_tensor = xywh_to_xyxy(bbox_tensor)
    bbox_tensor = torch.tensor(transform.apply_boxes(bbox_tensor.numpy(), original_size),
                               device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配
    rbox_tensor = torch.tensor(transform.apply_rboxes(rbox_tensor.numpy(), original_size),
                               device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        boxes=bbox_tensor,
        # boxes=bbox_tensor,
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

    # masks = sam.postprocess_masks(  # 对低分辨率掩码进行后处理，将其调整到输入图像的原始尺寸
    # low_res_masks,
    # input_size=input_size,  ##
    # original_size=original_size,
    # )

    return low_res_masks#, src

def update_logit_by_rend(original,result_rend):
    # 将 original 转换为 0 或 1
    original_binary = (original > 0.0).float()

    # 将 result_rend 转换为 0 或 1
    result_rend_binary = (result_rend > 0.0).float()

    # 找到二值化后不同的元素的掩码
    diff_mask = (original_binary != result_rend_binary).squeeze()
    # 将 original 内的这些元素取负值
    original[diff_mask] = -original[diff_mask]
    return original

from dataset import SingleJSONDataset_with_rbox


def train(epochs, batch_size, save_Path):
    training_logger = init_logger("training_sa1b_pointrend_2.log")
    train_avg_logger = init_logger("train_avg_sa1b_pointrend.log")
    val_avg_logger = init_logger("val_avg_sa1b_pointrend.log")
    test_avg_logger = init_logger("test_avg_sa1b_pointrend.log")
    # logging.basicConfig(filename='training_sa1b_pointrend.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
    checkpoint_path = r'./checkpoint_new/sam_conbine_epoch_88_5lr5.pth'  ##
    # checkpoint_path=None
    sam = init_model(checkpoint_path)
    sam.eval()
    #pointRend_checkpoint_path='./checkpoint_pointRend/pointRend_sam_epoch_4_lr4_origin_upsample_32_1dim.pth'
    pointRend_checkpoint_path = None
    pointRend = init_pointrend(pointRend_checkpoint_path)
    pos_random=PositionEmbeddingRandom(64)

    # 定义优化器，只优化可训练的参数
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, pointRend.parameters()), lr=1e-4, weight_decay=0.01)
    dice_loss = DiceLoss()
    focal_loss = FocalLoss()
    save_dir = save_Path
    best_iou = 0
    best_loss = 1000
    best_epoch = 0

    train_dataset_folder_path = r'/root/autodl-tmp/SA-1B/train-100'
    val_dataset_folder_path = r'/root/autodl-tmp/SA-1B/val-un10'
    train_image_files = [f for f in os.listdir(train_dataset_folder_path) if f.endswith('.jpg')]
    val_image_files = [f for f in os.listdir(val_dataset_folder_path) if f.endswith('.jpg')]
    num_train = 100
    num_val = 10
    
    if num_train>0:
        save_backup_file("train_pointrend_sa1b.py","lr4_origin_ConvTranspose2d_32_predicter_Linear124_2dim_norm_AllDropout2")
        save_backup_file("pointrend_test.py","lr4_origin_ConvTranspose2d_32_predicter_Linear124_2dim_norm_AllDropout2")

    # 划分数据集
    train_image_files = train_image_files[:num_train]
    val_image_files = val_image_files[:num_val]

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
        pointRend.train()
        total_train_loss = 0
        total_train_loss_1024 = 0
        total_train_loss_origin = 0
        total_train_dice_1024 = 0
        total_train_dice_origin = 0
        total_train_focal_1024 = 0
        total_train_focal_origin = 0
        # total_iou = 0
        total_num = 0
        loss = 0
        # 遍历所有图像文件
        for image_index, (image_filename, data) in tqdm(enumerate(zip(train_image_files, train_jsondata), start=1),total=len(train_image_files),desc=f'Training ==> Epoch:{epoch + 1}/{epochs}'):
            #if image_index<25:
                #continue
            dataset = SingleJSONDataset_with_rbox(data)
            dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True,num_workers=8)
            total_num += len(dataloader)
            image_file_path = os.path.join(train_dataset_folder_path, image_filename)
            image_embeddings = None
            for batch_idx, (bbox_tensor, rbox_tensor, gt_mask_tensor) in enumerate(dataloader,start=1):  # point_tensor:b12    gt_mask_tensor:bhw
                with torch.no_grad():
                    gt_mask_tensor = gt_mask_tensor.to(sam.device)
                    if image_embeddings is None:
                        image_embeddings, input_size, original_size, target_size = process_single_image(sam,image_file_path)
                    low_res_masks = process_single_mask(sam, image_embeddings, input_size, original_size, bbox_tensor,rbox_tensor)
                    low_res_masks = low_res_masks.detach()
                    #src=src.detach()

                #scaled_h, scaled_w = target_size  # max=1024
                #valid_h = scaled_h / input_size[0]  # 有效0-1
                #valid_w = scaled_w / input_size[1]  # 有效0-1

                # 1024
                pred_masks_1024 = F.interpolate(low_res_masks, (sam.image_encoder.img_size, sam.image_encoder.img_size),mode="bilinear", align_corners=False, )  # b 1 1024 1024
                pred_masks_1024 = pred_masks_1024[..., : input_size[0], : input_size[1]].contiguous()
                ##result_1024 = pointRend.forward_train(image_embeddings, pred_masks_1024, input_size[0], input_size[1])
                result_1024=0

                #result_rend_1024 = point_sample(pred_masks_1024.float(), result_1024["points"],align_corners=False)  # [B,N,1]
                
                #result_rend_1024 = point_sample_by_idx(pred_masks_1024, result_1024["idx"])  # [B,N,1]
                #result_rend_1024 = result_rend_1024 + result_1024["rend"]
                ##result_rend_1024 = result_1024["rend"]
                result_rend_1024=0
                
                ##gt_mask_tensor_1024 = (F.interpolate(gt_mask_tensor.unsqueeze(1), size=pred_masks_1024.shape[-2:], mode='bilinear')>0.0).float()  # b1hw
                gt_mask_tensor_1024=0
                #gt_points_1024 = point_sample(gt_mask_tensor_1024.float(), result_1024["points"], mode="nearest",align_corners=False).long()  # bnc
                ##gt_points_1024 = point_sample_by_idx(gt_mask_tensor_1024, result_1024["idx"]) # bnc
                gt_points_1024=0

                # 计算二元交叉熵损失
                ##loss_1024 = calculate_binary_cross_entropy_with_logits(result_rend_1024, gt_points_1024)
                ##dice_1024 = dice_loss(result_rend_1024 , (gt_points_1024 > 0.0).float())
                ##focal_1024 = focal_loss(result_rend_1024 ,  (gt_points_1024 > 0.0).float())
                loss_1024 =0
                dice_1024 =0
                focal_1024 =0
                
                # B, C, H, W = pred_masks_1024.shape
                # points_idx = result_1024["idx"].unsqueeze(1).expand(-1, C, -1)#b 1 n
                # pred_masks_1024 = (pred_masks_1024.reshape(B, C, -1)
                # .scatter_(2, points_idx, result_1024["rend"])
                # .view(B, C, H, W))
                
                """
                B, C, H, W = pred_masks_1024.shape
                pred_masks_1024=pred_masks_1024.view(B, -1)  # [B, H*W]
                for i in range(B):
                    original = torch.gather(
                        pred_masks_1024[i],
                        dim=0,
                        index=result_1024["idx"][i]
                    ).unsqueeze(-1)  # [N,1]
                    original = update_logit_by_rend(original,result_1024["rend"][i])  # <=n,1
                    pred_masks_1024[i].scatter_(dim=0, index=result_1024["idx"][i], src=original.squeeze(-1))
                pred_masks_1024=pred_masks_1024.view(B, C, H, W)
                """
                
                
            
                # origin
                pred_masks_origin = F.interpolate(pred_masks_1024, original_size, mode="bilinear", align_corners=False)
                result_origin = pointRend.forward_train(image_embeddings, pred_masks_origin, input_size[0], input_size[1],pos_random)
                ##result_origin=0
                
                #result_rend_origin = point_sample(pred_masks_origin.float(), result_origin["points"],align_corners=False)  # [B,N,1]
                
                #result_rend_origin = point_sample_by_idx(pred_masks_origin, result_origin["idx"])  # [B,N,1]
                #result_rend_origin = result_rend_origin + result_origin["rend"]
                result_rend_origin = result_origin["rend"]
                ##result_rend_origin=0
                
                #gt_points_origin = point_sample(gt_mask_tensor.float(), result_origin["points"], mode="nearest",align_corners=False).long()  # bnc
                gt_points_origin = point_sample_by_idx(gt_mask_tensor, result_origin["idx"])  # bnc
                ##gt_points_origin=0
                
                
                B, C, H, W = pred_masks_origin.shape
                pred_masks_origin=pred_masks_origin.view(B, -1)  # [B, H*W]
                for i in range(B):
                    original = torch.gather(
                        pred_masks_origin[i],
                        dim=0,
                        index=result_origin["idx"][i]
                    ).unsqueeze(-1)  # [N,1]
                    original = update_logit_by_rend(original,result_origin["rend"][i])  # <=n,1
                    pred_masks_origin[i].scatter_(dim=0, index=result_origin["idx"][i], src=original.squeeze(-1))
                pred_masks_origin=pred_masks_origin.view(B, C, H, W)
                
                
                # 计算二元交叉熵损失
                loss_origin = calculate_binary_cross_entropy_with_logits(result_rend_origin, gt_points_origin)
                
                #dice_origin = dice_loss(result_rend_origin , (gt_points_origin > 0.0).float())
                dice_origin = dice_loss(pred_masks_origin.squeeze(1) , (gt_mask_tensor > 0.0).float())
                
                focal_origin = focal_loss(result_rend_origin , (gt_points_origin > 0.0).float())    
                ##loss_origin=0
                ##dice_origin=0
                ##focal_origin=0
                
                #loss = 0.25 * loss_1024 + 0.25 * loss_origin+0.25*dice_1024+0.25*dice_origin+5*focal_1024+5*focal_origin
                loss =  loss_origin
                total_train_loss += loss.item()
                total_train_loss_1024 += loss_1024
                total_train_loss_origin += loss_origin
                total_train_dice_1024 += dice_1024
                total_train_dice_origin += dice_origin
                total_train_focal_1024 += focal_1024
                total_train_focal_origin += focal_origin
                
                # if batch_idx % 16 ==0 or batch_idx==len(dataloader):
                # 反向传播和优化
                optimizer.zero_grad(set_to_none=True)
                # print(f"start loss back {batch_idx}")
                loss.backward()
                
                # 计算梯度范数
                total_norm = 0
                for p in pointRend.parameters():
                    if p.grad is not None and p.requires_grad:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f'Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_train}  batch:{batch_idx}/{len(dataloader)}, Gradient norm = {total_norm}')
                
                optimizer.step()
                
                # 释放不必要的中间变量
                del pred_masks_1024, result_1024, result_rend_1024, gt_mask_tensor_1024, gt_points_1024
                del pred_masks_origin, result_origin, result_rend_origin, gt_points_origin
                torch.cuda.empty_cache()
                
                # loss=0
                # total_num+=1

                # print(f"finish loss back {batch_idx}")

                # B, C, H, W = pred_masks_origin.shape
                # points_idx = result_origin["idx"].unsqueeze(1).expand(-1, C, -1)#b 1 n
                # pred_masks_origin = (pred_masks_origin.reshape(B, C, -1)
                # .scatter_(2, points_idx, result_origin["rend"])
                # .view(B, C, H, W))

                # iou = calculate_iou(pred_masks_origin.squeeze(1) > 0.0, gt_mask_tensor)
                # total_iou += iou

                print(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_train}  batch:{batch_idx}/{len(dataloader)}, loss_1024: {loss_1024}, loss_origin: {loss_origin},dice_1024: {dice_1024}, dice_origin: {dice_origin},focal_1024: {focal_1024}, focal_origin: {focal_origin}, final_loss:{loss}')
                training_logger.info(f'TRAIN ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_train}  batch:{batch_idx}/{len(dataloader)}, loss_1024: {loss_1024}, loss_origin: {loss_origin},dice_1024: {dice_1024}, dice_origin: {dice_origin},focal_1024: {focal_1024}, focal_origin: {focal_origin}, final_loss:{loss}')

        avg_train_loss = total_train_loss / total_num if total_num > 0 else 0
        avg_train_loss_1024 = total_train_loss_1024 / total_num if total_num > 0 else 0
        avg_train_loss_origin = total_train_loss_origin / total_num if total_num > 0 else 0
        #avg_train_loss = total_train_loss / total_num if total_num > 0 else 0
        
        avg_train_dice_1024 = total_train_dice_1024 / total_num if total_num > 0 else 0
        avg_train_dice_origin = total_train_dice_origin / total_num if total_num > 0 else 0
        avg_train_focal_1024 = total_train_focal_1024 / total_num if total_num > 0 else 0
        avg_train_focal_origin = total_train_focal_origin / total_num if total_num > 0 else 0
        # avg_iou = total_iou / total_num
        print(
            f'TRAIN ==> Epoch {epoch + 1}/{epochs}, Train Loss_1024: {avg_train_loss_1024}, Train Loss_origin: {avg_train_loss_origin},Train avg_train_dice_1024: {avg_train_dice_1024},Train avg_train_dice_origin: {avg_train_dice_origin},Train avg_train_focal_1024: {avg_train_focal_1024},Train avg_train_focal_origin: {avg_train_focal_origin}, Train Loss: {avg_train_loss}')
        train_avg_logger.info(
            f'TRAIN ==> Epoch {epoch + 1}/{epochs}, Train Loss_1024: {avg_train_loss_1024}, Train Loss_origin: {avg_train_loss_origin},Train avg_train_dice_1024: {avg_train_dice_1024},Train avg_train_dice_origin: {avg_train_dice_origin},Train avg_train_focal_1024: {avg_train_focal_1024},Train avg_train_focal_origin: {avg_train_focal_origin}, Train Loss: {avg_train_loss}')

        # 验证阶段
        pointRend.eval()
        total_val_loss = 0
        total_val_loss_1024 = 0
        total_val_loss_origin = 0
        total_val_dice_1024 = 0
        total_val_dice_origin = 0
        total_val_focal_1024 = 0
        total_val_focal_origin = 0
        total_iou = 0
        total_num = 0
        with torch.no_grad():
            for image_index, (image_filename, data) in tqdm(enumerate(zip(val_image_files, val_jsondata), start=1),total=len(val_image_files),desc=f'Val ==> Epoch:{epoch + 1}/{epochs}'):
                dataset = SingleJSONDataset_with_rbox(data)
                dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True)
                total_num += len(dataloader)
                image_file_path = os.path.join(val_dataset_folder_path, image_filename)
                image_embeddings = None
                for batch_idx, (bbox_tensor, rbox_tensor, gt_mask_tensor) in enumerate(dataloader,start=1):  # point_tensor:b12    gt_mask_tensor:bhw
                    gt_mask_tensor = gt_mask_tensor.to(sam.device)
                    if image_embeddings is None:
                        image_embeddings, input_size, original_size, target_size = process_single_image(sam,image_file_path)
                    low_res_masks = process_single_mask(sam, image_embeddings, input_size, original_size, bbox_tensor,rbox_tensor)
                    low_res_masks = low_res_masks.detach()
                    #src=src.detach()

                    #scaled_h, scaled_w = target_size  # max=1024
                    #valid_h = scaled_h / input_size[0]  # 有效0-1
                    #valid_w = scaled_w / input_size[1]  # 有效0-1

                    # 1024
                    pred_masks_1024 = F.interpolate(low_res_masks,(sam.image_encoder.img_size, sam.image_encoder.img_size),mode="bilinear", align_corners=False, )  # b 1 1024 1024
                    pred_masks_1024 = pred_masks_1024[..., : input_size[0], : input_size[1]].contiguous()
                    result_1024 = pointRend.forward_eval(image_embeddings, pred_masks_1024, input_size[0], input_size[1],pos_random)
                    
                    gt_mask_tensor_1024 = (F.interpolate(gt_mask_tensor.unsqueeze(1), size=pred_masks_1024.shape[-2:], mode='bilinear')>0.0).float()  # b1hw
                    #gt_points_1024 = point_sample(gt_mask_tensor_1024.float(), result_1024["points"], mode="nearest",align_corners=False)  # list <=n,c
                    gt_points_1024 = point_sample_by_idx(gt_mask_tensor_1024, result_1024["idx"]) # list <=n,c
                    
                    ####################
                    # 获取原始mask的logits
                    B, C, H, W = pred_masks_1024.shape
                    flat_mask = pred_masks_1024.view(B, -1)  # [B, H*W]
                    #flat_mask2 = gt_mask_tensor_1024.view(B, -1)  # [B, H*W]
                    result_rend_1024 = []
                    for i in range(B):
                        original = torch.gather(
                            flat_mask[i],
                            dim=0,
                            index=result_1024["idx"][i]
                        ).unsqueeze(-1)  # [N,1]
                        #test_avg_logger.info(f'before 1024:{original}')
                        #test_avg_logger.info(f'result_1024:{result_1024["rend"][i]}')
                        #gt_original = torch.gather(
                            #flat_mask2[i],
                            #dim=0,
                            #index=result_1024["idx"][i]
                        #).unsqueeze(-1)  # [N,1]
                        #test_avg_logger.info(f'before gt_1024:{gt_original}')
                        #print(f"original 与真实结果相同的个数: {torch.sum((original > 0.0).float()== gt_original).item()}/{gt_original.shape[0]}")
                        #print(f"result_rend_1024 与真实结果相同的个数: {torch.sum((result_1024['rend'][i] > 0.0).float()== gt_original).item()}/{gt_original.shape[0]}")
                        
                        #original = result_1024["rend"][i]  # <=n,1
                        original = update_logit_by_rend(original,result_1024["rend"][i])  # <=n,1
                        
                        result_rend_1024.append(original)  # list <=n,1
                        #test_avg_logger.info(f'after 1024:{original}')
                        
                        # 替换 pred_masks_1024 中对应位置的值
                        flat_mask[i].scatter_(dim=0, index=result_1024["idx"][i], src=original.squeeze(-1))
                    #####################

                   
                    #test_avg_logger.info(f'gt_points_1024:{gt_points_1024}')
                    # 计算二元交叉熵损失
                    loss_1024 = calculate_binary_cross_entropy_with_logits(result_rend_1024, gt_points_1024)
                    dice_1024 = dice_loss(result_rend_1024 , gt_points_1024)
                    focal_1024 = focal_loss(result_rend_1024 ,  gt_points_1024)
                    # pred_masks_1024=update_masks(pred_masks_1024,result_1024)
                    pred_masks_1024 = flat_mask.view(B, C, H, W)

                    # origin
                    pred_masks_origin = F.interpolate(pred_masks_1024, original_size, mode="bilinear",align_corners=False)
                    result_origin = pointRend.forward_eval(image_embeddings, pred_masks_origin, input_size[0], input_size[1],pos_random)

                    ####################
                    # 获取原始mask的logits
                    B, C, H, W = pred_masks_origin.shape
                    flat_mask = pred_masks_origin.view(B, -1)  # [B, H*W]
                    #flat_mask2 = gt_mask_tensor.view(B, -1)  # [B, H*W]
                    result_rend_origin = []
                    for i in range(B):
                        original = torch.gather(
                            flat_mask[i],
                            dim=0,
                            index=result_origin["idx"][i]
                        ).unsqueeze(-1)  # [N,1]
                        #test_avg_logger.info(f'before origin:{original}')
                        #test_avg_logger.info(f'result_origin:{result_origin["rend"][i]}')
                        #gt_original = torch.gather(
                            #flat_mask2[i],
                            #dim=0,
                            #index=result_origin["idx"][i]
                        #).unsqueeze(-1)  # [N,1]
                        #test_avg_logger.info(f'before gt_origin:{gt_original}')
                        #print(f"original 与真实结果相同的个数: {torch.sum((original > 0.0).float()== gt_original).item()}/{gt_original.shape[0]}")
                        #print(f"result_rend_origin 与真实结果相同的个数: {torch.sum((result_origin['rend'][i] > 0.0).float()== gt_original).item()}/{gt_original.shape[0]}")
                        
                        #original =  result_origin["rend"][i]  # <=n,1
                       
                        original = update_logit_by_rend(original,result_origin["rend"][i])
                        
                        result_rend_origin.append(original)  # list <=n,1
                        #test_avg_logger.info(f'after origin:{original}')
                        # 替换 pred_masks_origin 中对应位置的值
                        flat_mask[i].scatter_(dim=0, index=result_origin["idx"][i], src=original.squeeze(-1))
                        #flat_mask[i].scatter_(dim=0, index=result_origin["idx"][i], src=1-gt_original.squeeze(-1))
                    #####################

                    #gt_points_origin = point_sample(gt_mask_tensor.float(), result_origin["points"], mode="nearest",align_corners=False)  # list <=n,c
                    gt_points_origin = point_sample_by_idx(gt_mask_tensor, result_origin["idx"])  # list <=n,c
                    #test_avg_logger.info(f'gt_points_origin:{gt_points_origin}')
                    
                    
                    pred_masks_origin = flat_mask.view(B, C, H, W)
                    
                    # 计算二元交叉熵损失
                    loss_origin = calculate_binary_cross_entropy_with_logits(result_rend_origin, gt_points_origin)
                    #dice_origin = dice_loss(result_rend_origin , gt_points_origin)
                    dice_origin = dice_loss(pred_masks_origin.squeeze(1) , (gt_mask_tensor > 0.0).float())
                    
                    focal_origin = focal_loss(result_rend_origin , gt_points_origin)  
                    
                    #loss = 0.25 * loss_1024 + 0.25 * loss_origin+0.25*dice_1024+0.25*dice_origin+5*focal_1024+5*focal_origin
                    loss =loss_origin
                    total_val_loss += loss.item()
                    total_val_loss_1024 += loss_1024
                    total_val_loss_origin += loss_origin
                    total_val_dice_1024 += dice_1024
                    total_val_dice_origin += dice_origin
                    total_val_focal_1024 += focal_1024
                    total_val_focal_origin += focal_origin
                    
                    
                    # pred_masks_origin=update_masks(pred_masks_origin,result_origin)
                    

                    iou = calculate_iou(pred_masks_origin.squeeze(1) > 0.0, gt_mask_tensor)
                    total_iou += iou
                    
                    # 释放不必要的中间变量
                    del pred_masks_1024, result_1024, result_rend_1024, gt_mask_tensor_1024, gt_points_1024
                    del pred_masks_origin, result_origin, result_rend_origin, gt_points_origin
                    torch.cuda.empty_cache()

                    print(f'VAL ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, loss_1024: {loss_1024}, loss_origin: {loss_origin},dice_1024: {dice_1024}, dice_origin: {dice_origin},focal_1024: {focal_1024}, focal_origin: {focal_origin},  final_loss:{loss}, iou:{iou}')
                    training_logger.info(f'VAL ==> Epoch {epoch + 1}/{epochs}, file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, loss_1024: {loss_1024}, loss_origin: {loss_origin},dice_1024: {dice_1024}, dice_origin: {dice_origin},focal_1024: {focal_1024}, focal_origin: {focal_origin},  final_loss:{loss}, iou:{iou}')

        avg_val_loss = total_val_loss / total_num
        avg_val_loss_1024 = total_val_loss_1024 / total_num
        avg_val_loss_origin = total_val_loss_origin / total_num
        avg_iou = total_iou / total_num
        avg_val_dice_1024 = total_val_dice_1024 / total_num if total_num > 0 else 0
        avg_val_dice_origin = total_val_dice_origin / total_num if total_num > 0 else 0
        avg_val_focal_1024 = total_val_focal_1024 / total_num if total_num > 0 else 0
        avg_val_focal_origin = total_val_focal_origin / total_num if total_num > 0 else 0
        print(f'VAL ==> Epoch {epoch + 1}/{epochs}, val Loss_1024: {avg_val_loss_1024}, val Loss_origin: {avg_val_loss_origin},Train avg_val_dice_1024: {avg_val_dice_1024},Train avg_val_dice_origin: {avg_val_dice_origin},Train avg_val_focal_1024: {avg_val_focal_1024},Train avg_val_focal_origin: {avg_val_focal_origin}, val Loss: {avg_val_loss} IoU: {avg_iou}')
        val_avg_logger.info(f'VAL ==> Epoch {epoch + 1}/{epochs}, val Loss_1024: {avg_val_loss_1024}, val Loss_origin: {avg_val_loss_origin},Train avg_val_dice_1024: {avg_val_dice_1024},Train avg_val_dice_origin: {avg_val_dice_origin},Train avg_val_focal_1024: {avg_val_focal_1024},Train avg_val_focal_origin: {avg_val_focal_origin}, val Loss: {avg_val_loss} IoU: {avg_iou}')
        
        if num_train>0:
            # 保存图像编码器的参数
            checkpoint_path = os.path.join(save_dir, f'pointRend_sam_epoch_{epoch + 1}_lr4_origin_ConvTranspose2d_32_predicter_Linear124_2dim_norm_AllDropout2.pth')
            torch.save(pointRend.state_dict(), checkpoint_path)
            print(f'Saved pointRend parameters to {checkpoint_path}')
            training_logger.info(f'Saved pointRend parameters to {checkpoint_path}')

            if avg_iou > best_iou:
                best_epoch = epoch + 1
                best_loss = avg_val_loss
                best_iou = avg_iou
                # 保存图像编码器的参数
                checkpoint_path = os.path.join(save_dir, f'pointRend_sam_best_epoch_lr4_origin_ConvTranspose2d_32_predicter_Linear124_2dim_norm_AllDropout2.pth')
                torch.save(pointRend.state_dict(), checkpoint_path)
                print(f'Saved pointRend parameters to {checkpoint_path}')
                training_logger.info(f'Saved pointRend parameters to {checkpoint_path}')

    print(f'训练已完成 Epoch：{epochs}, Batch: {batch_size}, num_train: {num_train}, num_val: {num_val}, best_epoch:{best_epoch}, best_loss:{best_loss}, best_iou:{best_iou} ')
    training_logger.info(f'训练已完成 Epoch：{epochs}, Batch: {batch_size}, num_train: {num_train}, num_val: {num_val}, best_epoch:{best_epoch}, best_loss:{best_loss}, best_iou:{best_iou} ')


# 获取当前脚本文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
save_Path = os.path.join(current_dir, "checkpoint_pointRend")
train(20, 16, save_Path)

