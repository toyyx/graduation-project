import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
# from mmdet.models.losses.giou_loss import GIoULoss
import torchvision.transforms as T
from PIL import Image
from scipy.optimize import linear_sum_assignment

from groundingdino.util import box_ops
from dataset.dataset import COCODataset_rbox
from groundingdino.models import build_model
from groundingdino.util.misc import nested_tensor_from_tensor_list
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging


def init_model(checkpoint_path=None):
    model_config_path = "./groundingdino/config/GroundingDINO_rbox_SwinT_OGC.py"
    ckpt_file_path = "./weights/groundingdino_swint_ogc.pth"
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    # args.device = 'cpu'

    if checkpoint_path is None:
        checkpoint = torch.load(ckpt_file_path, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(ckpt_file_path, log))
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        log = model.load_state_dict(checkpoint, strict=False)
        print("Model loaded from {} \n => {}".format(checkpoint_path, log))

    # 获取预训练权重中的参数名
    # pretrained_param_names = set(clean_state_dict(checkpoint['model']).keys())

    # _ = model.eval()

    # 遍历模型的所有参数
    for name, param in model.named_parameters():
        if 'fix_embed' not in name:
            # 如果参数在预训练权重中存在，则冻结该参数
            param.requires_grad_(False)
        else:
            # 如果参数在预训练权重中不存在，则设置为可训练
            param.requires_grad_(True)
            print(f"Trainable: {name}")  # 验证可训练参数
    return model


def init_dataloder(ann_file, image_dir, batch_size, max_data_size=None):
    def deal_batch_data(batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0])
        return tuple(batch)

    # 定义图像预处理操作
    transform = T.Compose([
        T.RandomResize([512], max_size=1333),  # 800
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = COCODataset_rbox(ann_file, image_dir, transform=transform, max_data_size=max_data_size)
    dataloder = DataLoader(dataset, batch_size=batch_size, collate_fn=deal_batch_data)
    return dataloder


def cxcywh_to_xyxy(boxes):
    """
    将 cxcywh 格式的边界框转换为 xyxy 格式
    :param boxes: 形状为 (N, 4) 的张量，N 是边界框数量，每个边界框是 [cx, cy, w, h] 格式
    :return: 形状为 (N, 4) 的张量，每个边界框是 [x1, y1, x2, y2] 格式
    """
    cx = boxes[:, 0]
    cy = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


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


# 计算广义交并比（GIoU），输入格式为 xyxy
def giou_matrix(boxes1, boxes2):
    """
    boxes1: [N, 4] (x1, y1, x2, y2)
    boxes2: [M, 4] (x1, y1, x2, y2)
    """
    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # 计算交集的坐标
    inter_x1 = torch.max(x1_1[:, None], x1_2[None, :])
    inter_y1 = torch.max(y1_1[:, None], y1_2[None, :])
    inter_x2 = torch.min(x2_1[:, None], x2_2[None, :])
    inter_y2 = torch.min(y2_1[:, None], y2_2[None, :])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # 计算并集的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1[:, None] + area2[None, :] - inter_area

    # 计算包含两个框的最小矩形的坐标
    enclosing_x1 = torch.min(x1_1[:, None], x1_2[None, :])
    enclosing_y1 = torch.min(y1_1[:, None], y1_2[None, :])
    enclosing_x2 = torch.max(x2_1[:, None], x2_2[None, :])
    enclosing_y2 = torch.max(y2_1[:, None], y2_2[None, :])

    enclosing_w = enclosing_x2 - enclosing_x1
    enclosing_h = enclosing_y2 - enclosing_y1
    enclosing_area = enclosing_w * enclosing_h

    # 计算GIoU
    giou_value = inter_area / (union_area + 1e-6) - (enclosing_area - union_area) / (enclosing_area + 1e-6)
    return 1 - giou_value


def giou_onedim(boxes1, boxes2):
    """
    boxes1: [N, 4] (x1, y1, x2, y2)
    boxes2: [N, 4] (x1, y1, x2, y2)
    """
    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # 计算交集的坐标
    inter_x1 = torch.max(x1_1, x1_2)
    inter_y1 = torch.max(y1_1, y1_2)
    inter_x2 = torch.min(x2_1, x2_2)
    inter_y2 = torch.min(y2_1, y2_2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # 计算并集的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    # 计算包含两个框的最小矩形的坐标
    enclosing_x1 = torch.min(x1_1, x1_2)
    enclosing_y1 = torch.min(y1_1, y1_2)
    enclosing_x2 = torch.max(x2_1, x2_2)
    enclosing_y2 = torch.max(y2_1, y2_2)

    enclosing_w = enclosing_x2 - enclosing_x1
    enclosing_h = enclosing_y2 - enclosing_y1
    enclosing_area = enclosing_w * enclosing_h

    # 计算GIoU
    giou_value = inter_area / union_area - (enclosing_area - union_area) / enclosing_area
    return 1 - giou_value


# 定义一个函数来计算成本矩阵
def calculate_cost_matrix(outputs, targets):
    pred_logits = outputs["pred_logits"].cpu()
    pred_boxes = outputs["pred_boxes"].cpu()
    pred_fix = outputs["pred_fix"].cpu()
    # pred_ratio = outputs["pred_ratio"].cpu()

    batch_size = pred_logits.size(0)
    num_queries = pred_logits.size(1)
    num_targets = [len(target["bbox"]) for target in targets]

    cost_matrices = []
    for b in range(batch_size):
        # cost_matrix = torch.zeros((num_queries, num_targets[b]), device=device)

        # 计算框成本
        pred_boxes_b = pred_boxes[b]
        target_boxes = targets[b]["bbox"]  # CxCywh

        # 这里简单使用 L1 距离作为框成本
        box_l1_cost = torch.cdist(pred_boxes_b, target_boxes, p=1)

        box_giou = giou_matrix(box_ops.box_cxcywh_to_xyxy(pred_boxes_b), box_ops.box_cxcywh_to_xyxy(target_boxes))

        # 综合成本
        cost_matrix = 5 * box_l1_cost + 2 * box_giou  # nm

        cost_matrices.append(cost_matrix)

    return cost_matrices


# 定义函数来获取每个批次的 row_indices, col_indices 对应关系
def get_matching_indices(cost_matrices):
    batch_size = len(cost_matrices)
    matching_indices_list = []
    for b in range(batch_size):
        cost_matrix = cost_matrices[b].cpu().detach().numpy()
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matching_indices_list.append((row_indices, col_indices))
    return matching_indices_list

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


def restore_boxes(target_boxes, width, height):
    target_boxes = np.array(target_boxes)
    cx = target_boxes[:, 0] * width.cpu().numpy()
    cy = target_boxes[:, 1] * height.cpu().numpy()
    w = target_boxes[:, 2] * width.cpu().numpy()
    h = target_boxes[:, 3] * height.cpu().numpy()
    result = np.column_stack((cx, cy, w, h))
    return result

# 定义函数来计算损失
def calculate_loss(outputs, targets, matching_indices_list):
    pred_logits = outputs["pred_logits"].cpu()
    pred_boxes = outputs["pred_boxes"].cpu()
    pred_fix = outputs["pred_fix"].cpu()
    # pred_ratio = outputs["pred_ratio"].cpu()

    batch_size = len(targets)
    bbox_l1_loss_list = []
    bbox_giou_loss_list = []
    fix_smooth_l1_loss_list = []
    ratio_smooth_l1_loss_list = []
    angle_loss_list = []

    for b in range(batch_size):
        row_indices, col_indices = matching_indices_list[b]
        target_boxes = targets[b]["bbox"] #cxcywh 0-1
        target_fix = targets[b]["fix"]
        # target_ratio =targets[b]["ratio"]
        h,w = targets[b]["size"]

        # 提取匹配的预测值
        matched_pred_boxes = pred_boxes[b][row_indices]
        matched_pred_fix = pred_fix[b][row_indices]
        # matched_pred_ratio = pred_ratio[b][row_indices]

        # 计算边界框的 L1 损失
        bbox_l1_loss = F.l1_loss(matched_pred_boxes, target_boxes[col_indices], reduction='sum') / matched_pred_boxes.shape[0]

        # 计算边界框的 GIoU 损失
        bbox_giou_loss = giou_onedim(box_ops.box_cxcywh_to_xyxy(matched_pred_boxes),box_ops.box_cxcywh_to_xyxy(target_boxes[col_indices])).mean()

        # 计算旋转顶点修正的 fix 的 SmoothL1 损失
        fix_smooth_l1_loss = F.smooth_l1_loss(matched_pred_fix, target_fix[col_indices], reduction='sum') / matched_pred_fix.shape[0]

        gt_angles=calculate_angle(restore_boxes(target_boxes,w,h),target_fix)
        pred_angles=calculate_angle(restore_boxes(matched_pred_boxes,w,h),matched_pred_fix)
        angle_loss=angular_cosine_loss(pred_angles, gt_angles)


        # 计算面积比例 ratio 的 SmoothL1 损失
        # ratio_smooth_l1_loss = F.smooth_l1_loss(matched_pred_ratio, target_ratio[col_indices],reduction='mean')

        # total_loss = bbox_l1_loss + bbox_giou_loss + fix_smooth_l1_loss + ratio_smooth_l1_loss
        # loss_list.append(total_loss)

        bbox_l1_loss_list.append(bbox_l1_loss)
        bbox_giou_loss_list.append(bbox_giou_loss)
        fix_smooth_l1_loss_list.append(fix_smooth_l1_loss)
        # ratio_smooth_l1_loss_list.append(ratio_smooth_l1_loss)
        angle_loss_list.append(angle_loss)

    bbox_l1_loss_mean = torch.mean(torch.stack(bbox_l1_loss_list))
    bbox_giou_loss_mean = torch.mean(torch.stack(bbox_giou_loss_list))
    fix_smooth_l1_loss_mean = torch.mean(torch.stack(fix_smooth_l1_loss_list))
    angle_loss_mean = torch.mean(torch.stack(angle_loss_list))
    # ratio_smooth_l1_loss_mean = torch.mean(torch.stack(ratio_smooth_l1_loss_list))


    # return bbox_l1_loss_mean, bbox_giou_loss_mean, fix_smooth_l1_loss_mean, ratio_smooth_l1_loss_mean
    return bbox_l1_loss_mean, bbox_giou_loss_mean, fix_smooth_l1_loss_mean, angle_loss_mean


def get_angle_from_rbox(rbox):
    """
    分批次计算多个 rbox 的方向角度（0 - 180 度）
    :param rboxes: 形状为 (4, 2) 的 numpy 数组，n 是 rbox 的数量
    :return: 形状为 (1,) 的 numpy 数组，包含每个 rbox 的方向角度
    """
    rbox = np.array(rbox)
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


def calculate_angle_from_rboxes_tensor(rboxes_tensor):  # bhw
    batch_size = rboxes_tensor.size(0)
    angles = []
    for i in range(batch_size):
        rbox = rboxes_tensor[i].tolist()
        angle = get_angle_from_rbox(rbox)
        angles.append(angle)
    return angles

import numpy as np

def get_rbox_from_bbox_with_fix(bbox,fix):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    widths=x2-x1
    heights=y2-y1

    top_point=[x1 + widths * fix[0], y1]
    right_point=[x2 , y1 + heights * fix[1]]
    bottom_point=[x2 - widths * fix[2], y2]
    left_point=[x1 , y2 - heights * fix[3]]

    return [top_point,right_point,bottom_point,left_point]


def generate_rbox(bbox, fix):#bbox n4 fix n4
    n = len(bbox)
    rboxes = []

    for i in range(n):
        # 提取当前 bbox 的中心坐标和宽高
        cx, cy, w, h = bbox[i]

        # 计算原始四边形的四个顶点坐标
        left = cx - w / 2
        right = cx + w / 2
        top = cy - h / 2
        bottom = cy + h / 2

        bbox_xyxy=[left,top,right,bottom]
        rbox=get_rbox_from_bbox_with_fix(bbox_xyxy,fix[i])#42
        rboxes.append(rbox)
    return rboxes#n42

def calculate_angle(bbox,fix):
    rboxes=generate_rbox(bbox,fix)
    angles=calculate_angle_from_rboxes_tensor(torch.as_tensor(rboxes))
    return angles

# 保存模型的函数
def save_model(model, save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")
    logging.info(f"Model saved at {model_path}")


def init_logger(filename):
    # 创建第一个日志记录器
    logger1 = logging.getLogger(filename)
    logger1.setLevel(logging.INFO)

    # 创建第一个日志记录器的文件处理器
    file_handler1 = logging.FileHandler(filename)
    file_handler1.setLevel(logging.INFO)

    # 创建第一个日志记录器的格式化器
    formatter1 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler1.setFormatter(formatter1)

    # 将处理器添加到第一个日志记录器
    logger1.addHandler(file_handler1)

    return logger1


train_ann_file = r'/root/autodl-tmp/coco2017/annotations/instances_train2017_with_rbox_merge.json'
train_image_dir = r'/root/autodl-tmp/coco2017/train2017'
val_ann_file = r'/root/autodl-tmp/coco2017/annotations/instances_val2017_with_rbox_merge.json'
val_image_dir = r'/root/autodl-tmp/coco2017/val2017'
#train_ann_file = r'C:\Users\yangxinyao\Downloads\coco2017\annotations\instances_train2017_with_rbox2_merge.json'
#train_image_dir = r'C:\Users\yangxinyao\Downloads\coco2017\train2017'
#val_ann_file = r'C:\Users\yangxinyao\Downloads\coco2017\annotations\instances_val2017_with_rbox2_merge.json'
#val_image_dir = r'C:\Users\yangxinyao\Downloads\coco2017\val2017'
epochs = 5
batch_size = 16
save_model_dir = r'./checkpoint'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = r'./checkpoint/model_epoch_18_lr5.pth'
#checkpoint_path=None
model = init_model(checkpoint_path)
model.to(device)

trainable_params = [param for param in model.parameters() if param.requires_grad]
optimizer = torch.optim.AdamW(
    trainable_params,  # 仅选择可训练参数
    lr=1e-6,
    # weight_decay=1e-4
)
train_dataloader = init_dataloder(train_ann_file, train_image_dir, batch_size, max_data_size=10000)
val_dataloader = init_dataloder(val_ann_file, val_image_dir, batch_size, max_data_size=1000)

best_val_loss = float('inf')
# 配置日志记录
# logging.basicConfig(filename='training.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

trainlogger = init_logger('training.log')
train_avg_logger = init_logger('train_avg.log')
val_avg_logger = init_logger('val_avg.log')

num_batch_train = len(train_dataloader)
num_batch_val = len(val_dataloader)

for epoch in range(epochs):
    model.train()
    running_train_loss = 0.0
    running_train_bbox_l1_loss = 0.0
    running_train_bbox_giou_loss = 0.0
    running_train_fix_smooth_l1_loss = 0.0
    running_train_angle_loss = 0.0

    for batch_idx, (samples, targets) in tqdm(enumerate(train_dataloader), total=num_batch_train,desc=f'Training ==> Epoch:{epoch + 1}/{epochs}'):
        # batch_start_time = time.time()

        # data_transfer_start = time.time()
        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        # data_transfer_end = time.time()
        # data_transfer_time = data_transfer_end - data_transfer_start

        # forward_start = time.time()
        outputs = model(samples, captions=captions)
        # forward_end = time.time()
        # forward_time = forward_end - forward_start

        # cost_matrix_start = time.time()
        # 计算成本矩阵
        cost_matrices = calculate_cost_matrix(outputs, targets)
        # cost_matrix_end = time.time()
        # cost_matrix_time = cost_matrix_end - cost_matrix_start

        # 获取匹配索引
        # matching_indices_start = time.time()
        # 获取每个批次的 row_indices, col_indices 对应关系
        matching_indices_list = get_matching_indices(cost_matrices)
        # matching_indices_end = time.time()
        # matching_indices_time = matching_indices_end - matching_indices_start

        # loss_calculation_start = time.time()
        # 计算损失
        # bbox_l1_loss_mean, bbox_giou_loss_mean, fix_smooth_l1_loss_mean, ratio_smooth_l1_loss_mean = calculate_loss(outputs, targets, matching_indices_list)
        bbox_l1_loss_mean, bbox_giou_loss_mean, fix_smooth_l1_loss_mean, angle_loss_mean = calculate_loss(outputs, targets, matching_indices_list)
        # loss_calculation_end = time.time()
        # loss_calculation_time = loss_calculation_end - loss_calculation_start

        # 计算总损失
        # total_loss_start = time.time()
        # 计算总损失
        # total_loss = 1 * bbox_l1_loss_mean + 1 * bbox_giou_loss_mean + 1 * fix_smooth_l1_loss_mean + 16 * ratio_smooth_l1_loss_mean
        total_loss = fix_smooth_l1_loss_mean + angle_loss_mean  ####
        # total_loss_end = time.time()
        # total_loss_time = total_loss_end - total_loss_start

        running_train_bbox_l1_loss += bbox_l1_loss_mean
        running_train_bbox_giou_loss += bbox_giou_loss_mean
        running_train_fix_smooth_l1_loss += fix_smooth_l1_loss_mean
        running_train_angle_loss += angle_loss_mean
        # running_val_ratio_smooth_l1_loss += ratio_smooth_l1_loss_mean
        running_train_loss += total_loss

        # backward_start = time.time()
        # 这里可以进行反向传播和参数更新
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        # backward_end = time.time()
        # backward_time = backward_end - backward_start

        # batch_end_time = time.time()
        # batch_total_time = batch_end_time - batch_start_time
        """
        print(
            f"Training ==> Epoch:{epoch + 1}/{epochs} Batch:{batch_idx + 1}/{num_batch_train} "
            f"bbox_l1_loss:{bbox_l1_loss_mean} bbox_giou_loss:{bbox_giou_loss_mean} "
            f"fix_smooth_l1_loss:{fix_smooth_l1_loss_mean} ratio_smooth_l1_loss:{ratio_smooth_l1_loss_mean} "
            f"Batch Loss: {total_loss} "
            f"Data Transfer Time: {data_transfer_time:.4f}s "
            f"Forward Time: {forward_time:.4f}s "
            f"Cost Matrix Time: {cost_matrix_time:.4f}s "
            f"Matching Indices Time: {matching_indices_time:.4f}s "
            f"Loss Calculation Time: {loss_calculation_time:.4f}s "
            f"Total Loss Time: {total_loss_time:.4f}s "
            f"Backward Time: {backward_time:.4f}s "
            f"Batch Total Time: {batch_total_time:.4f}s"
        )
        logging.info(
            f"Training ==> Epoch:{epoch + 1}/{epochs} Batch:{batch_idx + 1}/{num_batch_train} "
            f"bbox_l1_loss:{bbox_l1_loss_mean} bbox_giou_loss:{bbox_giou_loss_mean} "
            f"fix_smooth_l1_loss:{fix_smooth_l1_loss_mean} ratio_smooth_l1_loss:{ratio_smooth_l1_loss_mean} "
            f"Batch Loss: {total_loss} "
            f"Data Transfer Time: {data_transfer_time:.4f}s "
            f"Forward Time: {forward_time:.4f}s "
            f"Cost Matrix Time: {cost_matrix_time:.4f}s "
            f"Matching Indices Time: {matching_indices_time:.4f}s "
            f"Loss Calculation Time: {loss_calculation_time:.4f}s "
            f"Total Loss Time: {total_loss_time:.4f}s "
            f"Backward Time: {backward_time:.4f}s "
            f"Batch Total Time: {batch_total_time:.4f}s"
        )

        """

        print(
            f"Training ==> Epoch:{epoch + 1}/{epochs} Batch:{batch_idx + 1}/{num_batch_train} bbox_l1_loss:{bbox_l1_loss_mean} bbox_giou_loss:{bbox_giou_loss_mean} fix_smooth_l1_loss:{fix_smooth_l1_loss_mean} angle_loss_mean:{angle_loss_mean} Batch Loss: {total_loss}")
        trainlogger.info(
            f"Training ==> Epoch:{epoch + 1}/{epochs} Batch:{batch_idx + 1}/{num_batch_train} bbox_l1_loss:{bbox_l1_loss_mean} bbox_giou_loss:{bbox_giou_loss_mean} fix_smooth_l1_loss:{fix_smooth_l1_loss_mean} angle_loss_mean:{angle_loss_mean} Batch Loss: {total_loss}")

    # 计算验证集平均损失
    avg_train_bbox_l1_loss = running_train_bbox_l1_loss / num_batch_train
    avg_train_bbox_giou_loss = running_train_bbox_giou_loss / num_batch_train
    avg_train_fix_smooth_l1_loss = running_train_fix_smooth_l1_loss / num_batch_train
    avg_train_angle_loss = running_train_angle_loss / num_batch_train
    # avg_val_ratio_smooth_l1_loss = running_val_ratio_smooth_l1_loss / num_batch_val
    avg_train_loss = running_train_loss / num_batch_train
    print(
        f'Training_Average ==> Epoch:{epoch + 1}/{epochs} bbox_l1_loss:{avg_train_bbox_l1_loss} bbox_giou_loss:{avg_train_bbox_giou_loss} fix_smooth_l1_loss:{avg_train_fix_smooth_l1_loss} angle_loss_mean:{avg_train_angle_loss} Average Val Loss: {avg_train_loss}')
    train_avg_logger.info(
        f'Training_Average ==> Epoch:{epoch + 1}/{epochs} bbox_l1_loss:{avg_train_bbox_l1_loss} bbox_giou_loss:{avg_train_bbox_giou_loss} fix_smooth_l1_loss:{avg_train_fix_smooth_l1_loss} angle_loss_mean:{avg_train_angle_loss} Average Val Loss: {avg_train_loss}')

    # 在验证集上进行评估
    model.eval()
    running_val_loss = 0.0
    running_val_bbox_l1_loss = 0.0
    running_val_bbox_giou_loss = 0.0
    running_val_fix_smooth_l1_loss = 0.0
    running_val_angle_loss = 0.0
    # running_val_ratio_smooth_l1_loss = 0.0
    with torch.no_grad():
        for batch_idx, (samples, targets) in tqdm(enumerate(val_dataloader), total=num_batch_val,
                                                  desc=f'Validation ==> Epoch:{epoch + 1}/{epochs}'):
            samples = samples.to(device)
            captions = [t["caption"] for t in targets]
            outputs = model(samples, captions=captions)
            # 计算成本矩阵
            cost_matrices = calculate_cost_matrix(outputs, targets)
            # 获取每个批次的 row_indices, col_indices 对应关系
            matching_indices_list = get_matching_indices(cost_matrices)
            # 计算损失
            # bbox_l1_loss_mean, bbox_giou_loss_mean, fix_smooth_l1_loss_mean, ratio_smooth_l1_loss_mean = calculate_loss(outputs, targets, matching_indices_list)
            bbox_l1_loss_mean, bbox_giou_loss_mean, fix_smooth_l1_loss_mean, angle_loss_mean = calculate_loss(outputs, targets, matching_indices_list)

            # 计算总损失
            # total_loss = 1 * bbox_l1_loss_mean + 1 * bbox_giou_loss_mean + 1 * fix_smooth_l1_loss_mean + 16 * ratio_smooth_l1_loss_mean
            total_loss =  fix_smooth_l1_loss_mean + angle_loss_mean

            running_val_bbox_l1_loss += bbox_l1_loss_mean
            running_val_bbox_giou_loss += bbox_giou_loss_mean
            running_val_fix_smooth_l1_loss += fix_smooth_l1_loss_mean
            running_val_angle_loss += angle_loss_mean
            # running_val_ratio_smooth_l1_loss += ratio_smooth_l1_loss_mean
            running_val_loss += total_loss
            print(
                f"Validation ==> Epoch:{epoch + 1}/{epochs} Batch:{batch_idx + 1}/{num_batch_val} bbox_l1_loss:{bbox_l1_loss_mean} bbox_giou_loss:{bbox_giou_loss_mean} fix_smooth_l1_loss:{fix_smooth_l1_loss_mean} angle_loss_mean:{angle_loss_mean} Batch Loss: {total_loss}")
            trainlogger.info(
                f"Validation ==> Epoch:{epoch + 1}/{epochs} Batch:{batch_idx + 1}/{num_batch_val} bbox_l1_loss:{bbox_l1_loss_mean} bbox_giou_loss:{bbox_giou_loss_mean} fix_smooth_l1_loss:{fix_smooth_l1_loss_mean} angle_loss_mean:{angle_loss_mean} Batch Loss: {total_loss}")
    # 计算验证集平均损失
    avg_val_bbox_l1_loss = running_val_bbox_l1_loss / num_batch_val
    avg_val_bbox_giou_loss = running_val_bbox_giou_loss / num_batch_val
    avg_val_fix_smooth_l1_loss = running_val_fix_smooth_l1_loss / num_batch_val
    avg_val_angle_loss = running_val_angle_loss / num_batch_val
    # avg_val_ratio_smooth_l1_loss = running_val_ratio_smooth_l1_loss / num_batch_val
    avg_val_loss = running_val_loss / num_batch_val
    print(
        f'Validation_Average ==> Epoch:{epoch + 1}/{epochs} bbox_l1_loss:{avg_val_bbox_l1_loss} bbox_giou_loss:{avg_val_bbox_giou_loss} fix_smooth_l1_loss:{avg_val_fix_smooth_l1_loss} angle_loss_mean:{avg_val_angle_loss} Average Val Loss: {avg_val_loss}')
    val_avg_logger.info(
        f'Validation_Average ==> Epoch:{epoch + 1}/{epochs} bbox_l1_loss:{avg_val_bbox_l1_loss} bbox_giou_loss:{avg_val_bbox_giou_loss} fix_smooth_l1_loss:{avg_val_fix_smooth_l1_loss} angle_loss_mean:{avg_val_angle_loss} Average Val Loss: {avg_val_loss}')
    save_model(model, save_model_dir, f"model_epoch_{epoch + 1 + 18}_lr6.pth")

    # 根据验证集损失保存最佳模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_model(model, save_model_dir, f"model_best_lr6.pth")