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
from dataset_sa1b import SingleJSONDataset
from dataset_sa1b import SingleJSONDataset_with_rbox,SingleJSONDataset_with_bbox
from torchvision.transforms.functional import gaussian_blur
import numpy as np

import sys
import os
other_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(other_folder_path)

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model=build_efficient_sam_vitt()
    model.eval()
    return model.to(device)
    #return build_efficient_sam_vits()

def process_single_image(sam, image_file_path):
    sample_image_np = np.array(Image.open(image_file_path))
    sample_image_tensor = transforms.ToTensor()(sample_image_np)[None, ...].to(sam.device)#1,???

    batch_size, _, input_h, input_w = sample_image_tensor.shape
    image_embeddings = sam.get_image_embeddings(sample_image_tensor)

    return image_embeddings,input_h, input_w

def process_single_mask(sam, image_embeddings,input_h,input_w, bbox_tensor, rbox_tensor):
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
    b=bbox_tensor.shape[0]
    bbox_tensor = xywh_to_xyxy(bbox_tensor)#b，4
    bbox_tensor = bbox_tensor.view(b,1,2,2).to(sam.device)#b,1,2,2
    bbox_label = torch.tensor([2, 3]).expand(b, 1, 2).to(sam.device)#b,1,2

    #b,1,1,h,w
    masks,_=sam.predict_masks(
        image_embeddings,
        bbox_tensor,
        bbox_label,
        multimask_output=True,
        input_h=input_h,
        input_w=input_w,
        output_h=input_h,
        output_w=input_w,
    )
    return masks.squeeze(1)[:,:1,:,:]#b,1,h,w

from dataset_lvis import LVISFullDataset, LVISAnnotationDataset
from lvis import LVIS

# 自定义 collate_fn 函数，直接返回原始数据
def custom_collate(batch):
    #print(batch)
    data=batch[0]
    return data[0],data[1],data[2]
def evaluate(batch_size, save_Path):
    logging.basicConfig(filename='evaluate_sa1b_combine.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    sam = init_model()

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    # dice_loss = DBSLoss()

    lvis_json_path = r'/root/autodl-tmp/lvis_v1_val.json'
    coco_image_root_path = r'/root/autodl-tmp/coco2017'

    lvis_api = LVIS(annotation_path=lvis_json_path)

    num_val = None
    image_dataset = LVISFullDataset(lvis_api, coco_image_root_path, max=num_val)
    image_dataloader = DataLoader(image_dataset, batch_size=1, collate_fn=custom_collate)
    num_val = len(image_dataloader)

    # 验证阶段
    sam.eval()
    total_val_loss = 0
    total_accuracy = 0
    total_iou = 0
    total_num = 0
    with torch.no_grad():
        # 遍历文件夹中的所有文件
        for image_index, (image_path, img_info, anns) in tqdm(enumerate(image_dataloader, start=1),total=len(image_dataloader), desc=f'Evaluate ==> '):
            dataset = LVISAnnotationDataset(lvis_api, img_info, anns)
            dataloader = DataLoader(dataset, batch_size=batch_size)
            total_num += len(dataloader)
            image_embeddings = None
            for batch_idx, (bbox_tensor, rbox_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):  # point_tensor:b12    gt_mask_tensor:bhw
                if image_embeddings is None:
                    image_embeddings, input_h, input_w = process_single_image(sam, image_path)####
                masks = process_single_mask(sam, image_embeddings, input_h, input_w, bbox_tensor, rbox_tensor).cpu()  # bchw
                # 计算损失
                focal = focal_loss(masks.squeeze(1), gt_mask_tensor)
                dice = dice_loss(masks.squeeze(1), gt_mask_tensor)
                loss = 20 * focal + dice
                total_val_loss += loss.item()

                accuracy = calculate_accuracy(masks.squeeze(1) > 0.0, gt_mask_tensor)
                total_accuracy += accuracy

                iou = calculate_iou(masks.squeeze(1) > 0.0, gt_mask_tensor)
                total_iou += iou
                print(f'VAL ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')
                logging.info(f'VAL ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, focal_loss: {focal}, dice_loss: {dice}, final_loss:{loss}, accuracy:{accuracy}, iou:{iou}')

    avg_val_loss = total_val_loss / total_num
    avg_accuracy = total_accuracy / total_num
    avg_iou = total_iou / total_num
    print(f'VAL ==> Val Loss: {avg_val_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')
    logging.info(f'VAL ==> Val Loss: {avg_val_loss}, Accuracy: {avg_accuracy}, IoU: {avg_iou}')


# 获取当前脚本文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
save_Path = os.path.join(current_dir, "checkpoint_new")
evaluate(8, save_Path)

