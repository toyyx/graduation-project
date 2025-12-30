import logging
import math
import os
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib
# 修改后端为 Agg，适合在无图形界面的环境中保存图片
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from build_sam import sam_model_registry
from typing import Any, Optional, Tuple, Type
import numpy as np
from utils.transforms import ResizeLongestSide
from torchvision.transforms.functional import gaussian_blur


# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


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
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # checkpoint_path = os.path.join(current_dir, "checkpoint_bb","sam_epoch_3.pth")

    # 加载SAM模型
    sam = sam_model_registry[model_type]()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam.to(device)

    if checkpoint_path is None:
        checkpoint = torch.load(sam_checkpoint_path, map_location=device)

        log = sam.load_state_dict(checkpoint, strict=False)

        print("Model loaded from {} \n => {}".format(sam_checkpoint_path, log))

        # sam.prompt_encoder.rbox_point_embeddings[0].weight.data = sam.prompt_encoder.point_embeddings[2].weight.data.clone()
        # sam.prompt_encoder.rbox_point_embeddings[1].weight.data = sam.prompt_encoder.point_embeddings[3].weight.data.clone()
        # sam.prompt_encoder.rbox_point_embeddings[2].weight.data = sam.prompt_encoder.point_embeddings[3].weight.data.clone()
        # sam.prompt_encoder.rbox_point_embeddings[3].weight.data = sam.prompt_encoder.point_embeddings[2].weight.data.clone()
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        log = sam.load_state_dict(checkpoint, strict=False)

        print("Model loaded from {} \n => {}".format(checkpoint_path, log))

    # 提取提示编码器和掩码解码器的权重
    # image_encoder_weights = {k.replace('image_encoder.', ''): v for k, v in checkpoint.items() if k.startswith('image_encoder.')}
    # prompt_encoder_weights = {k.replace('prompt_encoder.', ''): v for k, v in checkpoint.items() if k.startswith('prompt_encoder.')}
    # mask_decoder_weights = {k.replace('mask_decoder.', ''): v for k, v in checkpoint.items() if k.startswith('mask_decoder.')}

    # 加载提示编码器和掩码解码器的权重
    # sam.image_encoder.load_state_dict(image_encoder_weights)
    # sam.prompt_encoder.load_state_dict(prompt_encoder_weights)
    # sam.mask_decoder.load_state_dict(mask_decoder_weights)

    """
    # 冻结原有参数
    for name, param in sam.named_parameters():
        if 'rbox_point_embeddings' not in name:
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

    bbox_tensor = xywh_to_xyxy(bbox_tensor)
    bbox_tensor = torch.tensor(transform.apply_boxes(bbox_tensor.numpy(), original_size),
                               device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配
    rbox_tensor = torch.tensor(transform.apply_rboxes(rbox_tensor.numpy(), original_size),
                               device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        # boxes=None,
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


def process_single_bbox(sam, image_file_path, bbox_tensor):
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

    bbox_tensor = xywh_to_xyxy(bbox_tensor)
    bbox_tensor = torch.tensor(transform.apply_boxes(bbox_tensor.numpy(), original_size),
                               device=sam.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        boxes=bbox_tensor,
        rboxes=None,
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
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :,
                        :]  # 调整张量的维度顺序为 BCHW（批次、通道、高度、宽度），并确保内存连续

    original_size = image.shape[:2]  # 保存原始图像的尺寸
    input_size = tuple(input_image_torch.shape[-2:])  # 保存输入图像（变换后）的尺寸
    input_image = sam.preprocess(input_image_torch)  # 对输入图像进行预处理，如归一化、填充等操作
    image_embeddings = sam.image_encoder(input_image)

    return image_embeddings, input_size, original_size


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
        # boxes=None,
        boxes=bbox_tensor,
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
def process_single_mask_pointrend(sam, image_embeddings, input_size, original_size, bbox_tensor, rbox_tensor):
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
        # boxes=None,
        boxes=bbox_tensor,
        boxes_origin=None,
        rboxes=None,
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

    return low_res_masks

def process_single_mask_bbox(sam, image_embeddings, input_size, original_size, bbox_tensor):
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

    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
        points=None,
        # boxes=None,
        boxes=bbox_tensor,
        boxes_origin=None,
        rboxes=None,
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


from pointrend_test import PointRend, PointHead, point_sample, StandardPointHead, PointHead_try, PointRend_try, point_sample_by_idx

def init_pointrend(checkpoint_path=None):
    pointRend = PointRend_try(head=StandardPointHead(), training=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pointRend.to(device)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        log = pointRend.load_state_dict(checkpoint, strict=False)

        print("Model loaded from {} \n => {}".format(checkpoint_path, log))

    return pointRend

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


class PositionEmbeddingRandom():
    def __init__(self, num_pos_feats: int = 32, scale: Optional[float] = 1.0) -> None:
        self.positional_encoding_gaussian_matrix = scale * torch.randn((2, num_pos_feats // 2),
                                                                       device='cuda' if torch.cuda.is_available() else 'cpu')  # 注册一个缓冲区，用于存储随机生成的高斯矩阵

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
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
            return torch.cat([torch.sin(coords), torch.cos(coords)],
                             dim=-1)  # torch.Tensor: 位置编码后的张量，形状为 d_1 x ... x d_n x C

    def compute_pe_for_sampled_points(self, coords):
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


from skimage.measure import label


def keep_largest_connected_region(mask):
    """
    保留输入掩码中每个样本的最大连通区域，并将输出格式调整为 (b, 1, h, w) 的 torch.Tensor
    :param mask: 输入的掩码，形状为 (b, h, w)，可以是 numpy.ndarray 或 torch.Tensor
    :return: 仅保留最大连通区域后的掩码，形状为 (b, 1, h, w) 的 torch.Tensor
    """
    if isinstance(mask, torch.Tensor):
        device = mask.device
        mask = mask.cpu().numpy()
    else:
        device = torch.device('cpu')

    result = np.zeros((mask.shape[0], 1, mask.shape[1], mask.shape[2]), dtype=np.uint8)
    for i in range(mask.shape[0]):
        current_mask = mask[i]
        # 标记连通区域
        labeled_mask = label(current_mask)
        # 获取所有非零的标签
        unique_labels = np.unique(labeled_mask)
        unique_labels = unique_labels[unique_labels != 0]

        if len(unique_labels) == 0:
            continue

        # 计算每个连通区域的面积
        areas = [np.sum(labeled_mask == label) for label in unique_labels]
        # 找到最大连通区域的标签
        largest_label = unique_labels[np.argmax(areas)]
        # 生成仅包含最大连通区域的掩码
        largest_region_mask = (labeled_mask == largest_label).astype(np.uint8)
        result[i, 0] = largest_region_mask

    result = torch.from_numpy(result).to(device)
    return result


from dataset_lvis import LVISFullDataset, LVISAnnotationDataset
from lvis import LVIS


# 自定义 collate_fn 函数，直接返回原始数据
def custom_collate(batch):
    # print(batch)
    data = batch[0]
    return data[0], data[1], data[2]


def evaluate(batch_size):
    logging.basicConfig(filename='evaluate_lvis_sam.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    checkpoint_path = r'./checkpoint_new/sam_conbine_epoch_88_5lr5.pth'
    # checkpoint_path=None
    sam = init_model(checkpoint_path)

    pointRend_checkpoint_path = './checkpoint_pointRend/pointRend_sam_epoch_3_lr4_origin_ConvTranspose2d_32_predicter_Linear124_2dim_norm_dropout2.pth'
    # pointRend_checkpoint_path = None
    pointRend = init_pointrend(pointRend_checkpoint_path)
    pos_random = PositionEmbeddingRandom(64)

    lvis_json_path = r'/root/autodl-tmp/lvis_v1_val.json'
    coco_image_root_path = r'/root/autodl-tmp/coco2017'

    lvis_api = LVIS(annotation_path=lvis_json_path)

    num_val = None
    image_dataset = LVISFullDataset(lvis_api, coco_image_root_path, max=num_val)
    image_dataloader = DataLoader(image_dataset, batch_size=1, collate_fn=custom_collate)
    num_val = len(image_dataloader)

    # 验证阶段
    sam.eval()
    pointRend.eval()
    total_iou = 0
    total_num = 0
    with torch.no_grad():
        # 遍历文件夹中的所有文件 full_img_path, img_info, anns
        for image_index, (image_path, img_info, anns) in tqdm(enumerate(image_dataloader, start=1),
                                                              total=len(image_dataloader), desc=f'Evaluate ==> '):
            dataset = LVISAnnotationDataset(lvis_api, img_info, anns)
            dataloader = DataLoader(dataset, batch_size=batch_size)
            total_num += len(dataloader)
            image_embeddings = None
            for batch_idx, (bbox_tensor, rbox_tensor, gt_mask_tensor) in enumerate(dataloader,
                                                                                   start=1):  # point_tensor:b12    gt_mask_tensor:bhw
                gt_mask_tensor = gt_mask_tensor.to(sam.device)
                if image_embeddings is None:
                    image_embeddings, input_size, original_size = process_single_image(sam, image_path)

                low_res_masks = process_single_mask_pointrend(sam, image_embeddings, input_size, original_size,
                                                              bbox_tensor, rbox_tensor)

                # 1024
                pred_masks_1024 = F.interpolate(low_res_masks, (sam.image_encoder.img_size, sam.image_encoder.img_size),
                                                mode="bilinear", align_corners=False, )  # b 1 1024 1024
                pred_masks_1024 = pred_masks_1024[..., : input_size[0], : input_size[1]].contiguous()
                result_1024 = pointRend.forward_eval(image_embeddings, pred_masks_1024, input_size[0], input_size[1],
                                                     pos_random)

                ####################
                # 获取原始mask的logits
                B, C, H, W = pred_masks_1024.shape
                flat_mask = pred_masks_1024.view(B, -1)  # [B, H*W]
                result_rend_1024 = []
                for i in range(B):
                    original = torch.gather(
                        flat_mask[i],
                        dim=0,
                        index=result_1024["idx"][i]
                    ).unsqueeze(-1)  # [N,1]
                    original = update_logit_by_rend(original, result_1024["rend"][i])  # <=n,1
                    result_rend_1024.append(original)  # list <=n,1
                    # 替换 pred_masks_1024 中对应位置的值
                    flat_mask[i].scatter_(dim=0, index=result_1024["idx"][i], src=original.squeeze(-1))
                #####################
                # 计算二元交叉熵损失
                pred_masks_1024 = flat_mask.view(B, C, H, W)

                # origin
                pred_masks_origin = F.interpolate(pred_masks_1024, original_size, mode="bilinear", align_corners=False)
                result_origin = pointRend.forward_eval(image_embeddings, pred_masks_origin, input_size[0],
                                                       input_size[1], pos_random)
                ####################
                # 获取原始mask的logits
                B, C, H, W = pred_masks_origin.shape
                flat_mask = pred_masks_origin.view(B, -1)  # [B, H*W]
                result_rend_origin = []
                for i in range(B):
                    original = torch.gather(
                        flat_mask[i],
                        dim=0,
                        index=result_origin["idx"][i]
                    ).unsqueeze(-1)  # [N,1]
                    original = update_logit_by_rend(original, result_origin["rend"][i])
                    result_rend_origin.append(original)  # list <=n,1
                    flat_mask[i].scatter_(dim=0, index=result_origin["idx"][i], src=original.squeeze(-1))
                #####################
                pred_masks_origin = flat_mask.view(B, C, H, W)

                iou = calculate_iou(pred_masks_origin.squeeze(1) > 0.0, gt_mask_tensor)
                total_iou += iou

                print(
                    f'Evaluate ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, iou:{iou}')
                logging.info(
                    f'Evaluate ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, iou:{iou}')
                """
                # 可视化部分
                image = cv2.imread(image_path[0])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pred_mask = masks.squeeze(1)[0].numpy() > 0.0
                gt_mask = gt_mask_tensor[0].numpy()
                bbox = bbox_tensor[0].numpy().astype(int)

                fig, axes = plt.subplots(1, 5, figsize=(20, 5))

                # 显示原始图片
                axes[0].imshow(image)
                axes[0].set_title('Original Image')

                # 显示预测掩码
                axes[1].imshow(image)
                axes[1].imshow(pred_mask, alpha=0.5, cmap='jet')
                axes[1].set_title('Predicted Mask')

                # 显示真实掩码
                axes[2].imshow(image)
                axes[2].imshow(gt_mask, alpha=0.5, cmap='jet')
                axes[2].set_title('Ground Truth Mask')

                # 显示带有边界框的图片
                x, y, w, h = bbox
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                axes[3].imshow(image)
                axes[3].set_title('Image with Bounding Box')

                # 显示带有 rbox_tensor 的图片
                rbox_points = rbox_tensor[0].numpy().astype(int)
                for i in range(4):
                    axes[4].plot(rbox_points[i, 0], rbox_points[i, 1], 'ro')  # 用红色圆点绘制点
                    if i < 3:
                        axes[4].plot([rbox_points[i, 0], rbox_points[i + 1, 0]],
                                     [rbox_points[i, 1], rbox_points[i + 1, 1]], 'r-')  # 连接点
                    else:
                        axes[4].plot([rbox_points[i, 0], rbox_points[0, 0]], [rbox_points[i, 1], rbox_points[0, 1]],
                                     'r-')  # 连接最后一点和第一点
                axes[4].imshow(image)
                axes[4].set_title('Image with Rotated Bounding Box')

                plt.show()
                """


    avg_iou = total_iou / total_num
    print(f'Evaluate ==>  IoU: {avg_iou}')
    logging.info(f'Evaluate ==>  IoU: {avg_iou}')



evaluate(batch_size=8)

