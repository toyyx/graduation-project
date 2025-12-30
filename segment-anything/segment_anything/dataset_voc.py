import os
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
import numpy as np


class PascalVOCImagePathDataset(Dataset):
    def __init__(self, txt_file_path, image_dir, instance_seg_dir, annotation_dir,num_max_image=None):
        self.image_names = []
        with open(txt_file_path, 'r') as f:
            for line in f:
                self.image_names.append(line.strip())
        if num_max_image is not None and num_max_image <= len(self.image_names):
            self.image_names = self.image_names[:num_max_image]
        self.image_dir = image_dir
        self.instance_seg_dir = instance_seg_dir
        self.annotation_dir = annotation_dir

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, f"{image_name}.jpg")
        instance_seg_path = os.path.join(self.instance_seg_dir, f"{image_name}.png")
        annotation_path = os.path.join(self.annotation_dir, f"{image_name}.xml")
        return image_path, instance_seg_path, annotation_path

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

def calculate_bbox_and_fix(rbox):
    """
    计算最小水平外接框、偏移比值和面积比值
    :param rbox: 旋转包围框的四个点坐标
    :return: new_box, fix, ratio
    """
    rbox = np.array(rbox)
    x_min = np.min(rbox[:, 0])
    y_min = np.min(rbox[:, 1])
    x_max = np.max(rbox[:, 0])
    y_max = np.max(rbox[:, 1])
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    new_bbox_xyxy = [x_min, y_min, x_max, y_max]

    # 按位置分为 top、right、bottom、left
    top_point = rbox[0]
    right_point = rbox[1]
    bottom_point = rbox[2]
    left_point = rbox[3]

    # 计算偏移比值
    fix_top = (top_point[0] - x_min) / bbox_width if bbox_width > 0 else 0
    fix_right = (right_point[1] - y_min) / bbox_height if bbox_height > 0 else 0
    fix_bottom = (x_max - bottom_point[0]) / bbox_width if bbox_width > 0 else 0
    fix_left = (y_max - left_point[1]) / bbox_height if bbox_height > 0 else 0
    fix = [fix_top, fix_right, fix_bottom, fix_left]

    return new_bbox_xyxy, fix

def get_rbox_from_bbox_with_fix(bbox,fix):
    x1 = bbox[0]
    y1 = bbox[1]
    widths = bbox[2]
    heights = bbox[3]
    x2 = x1 + widths
    y2 = y1 + heights

    top_point=[x1 + widths * fix[0], y1]
    right_point=[x2 , y1 + heights * fix[1]]
    bottom_point=[x2 - widths * fix[2], y2]
    left_point=[x1 , y2 - heights * fix[3]]

    return [top_point,right_point,bottom_point,left_point]

def get_bbox_from_mask(mask):
    """
    该函数用于从二值掩码中获取完整包住它的最小边界框
    :param mask: 输入的二值掩码，是一个二维的 NumPy 数组
    :return: 边界框的坐标 (x_min, y_min, x_max, y_max)
    """
    # 找出掩码中所有非零像素的坐标
    rows, columns = np.nonzero(mask)
    if len(rows) == 0 or len(columns) == 0:
        return None  # 如果掩码中没有非零像素，返回 None
    # 计算最小和最大的行坐标与列坐标
    y_min = np.min(rows)
    y_max = np.max(rows)
    x_min = np.min(columns)
    x_max = np.max(columns)
    return [x_min, y_min, x_max-x_min, y_max-y_min]


class PascalVOCInstanceSegDataset(Dataset):
    def __init__(self, instance_seg_path):
        self.instance_seg_path = instance_seg_path
        self.masks = []
        self.bboxes = []
        # 读取实例分割图像
        instance_seg_img = cv2.imread(instance_seg_path, cv2.IMREAD_GRAYSCALE)
        # 获取图像的高度和宽度
        self.height, self.width = instance_seg_img.shape[:2]
        # 提取所有实例的像素值
        instance_values = np.unique(instance_seg_img)

        # 过滤掉背景和边框像素值
        instance_values = [val for val in instance_values if val not in [0, 220]]
        for val in instance_values:
            # 生成对应的掩码
            mask = (instance_seg_img == val).astype(np.uint8)
            self.masks.append(mask)

            bbox = get_bbox_from_mask(mask)
            if bbox:
                self.bboxes.append(bbox)
            else:
                self.bboxes.append([0, 0, 0, 0])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        binary_mask = self.masks[idx]
        mask_tensor = torch.tensor(binary_mask, dtype=torch.float32)  # h * w

        bbox = self.bboxes[idx]
        bbox_tensor = torch.as_tensor(bbox, dtype=torch.float32)  # 4 xywh

        rbox = get_rbox_from_binary_mask(binary_mask,self.width,self.height)
        new_bbox_xyxy, fix = calculate_bbox_and_fix(rbox)
        rbox = get_rbox_from_bbox_with_fix(bbox, fix)
        rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)

        return bbox_tensor, rbox_tensor, mask_tensor

