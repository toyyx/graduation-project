import os

import cv2
import numpy as np
import torch
from lvis import LVIS
from torch.utils.data import Dataset


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


def get_rbox_from_bbox_with_fix(bbox, fix):
    x1 = bbox[0]
    y1 = bbox[1]
    widths = bbox[2]
    heights = bbox[3]
    x2 = x1 + widths
    y2 = y1 + heights

    top_point = [x1 + widths * fix[0], y1]
    right_point = [x2, y1 + heights * fix[1]]
    bottom_point = [x2 - widths * fix[2], y2]
    left_point = [x1, y2 - heights * fix[3]]

    return [top_point, right_point, bottom_point, left_point]


class LVISFullDataset(Dataset):
    def __init__(self, lvis_api, image_root_path, max=None):
        """
        初始化 LVISFullDataset 类
        :param json_path: LVIS 标注文件的 JSON 路径
        :param image_root_path: 图片的根路径
        """
        self.lvis_api = lvis_api
        self.img_ids = self.lvis_api.get_img_ids()
        self.image_root_path = image_root_path

        if max is not None and max <= len(self.img_ids):
            self.img_ids = self.img_ids[:max]

    def __len__(self):
        """
        返回数据集的长度
        :return: 数据集中图片的数量
        """
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        根据索引获取图片路径和对应的所有标注
        :param idx: 索引
        :return: 图片路径和对应的所有标注
        """
        img_id = self.img_ids[idx]
        img_info = self.lvis_api.load_imgs([img_id])[0]

        # 从 coco_url 中提取图片路径
        coco_url = img_info['coco_url']
        parts = coco_url.split('/')
        relative_img_path = '/'.join(parts[-2:])
        full_img_path = os.path.join(self.image_root_path, relative_img_path)

        # 获取该图像的所有标注
        ann_ids = self.lvis_api.get_ann_ids(img_ids=[img_id])
        anns = self.lvis_api.load_anns(ann_ids)

        return full_img_path, img_info, anns


class LVISAnnotationDataset(Dataset):
    def __init__(self, lvis_api, img_info, annotations):
        """
        初始化 LVISAnnotationDataset 类
        :param annotations: 一张图片对应的所有标注
        """
        self.lvis_api = lvis_api
        self.img_info = img_info
        self.annotations = annotations

    def __len__(self):
        """
        返回标注的数量
        :return: 标注的数量
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        根据索引获取 mask 和 bbox
        :param idx: 索引
        :return: mask 和 bbox
        """
        ann = self.annotations[idx]
        binary_mask = self.lvis_api.ann_to_mask(ann)
        mask_tensor = torch.tensor(binary_mask, dtype=torch.float32)  # h * w

        # old
        # bbox = np.array(ann['dino_bbox'], dtype=np.float32)
        bbox_tensor = torch.as_tensor(ann["bbox"], dtype=torch.float32)  # xywh

        rbox = get_rbox_from_binary_mask(binary_mask, self.img_info['width'], self.img_info['height'])
        new_bbox_xyxy, fix = calculate_bbox_and_fix(rbox)
        rbox = get_rbox_from_bbox_with_fix(ann["bbox"], fix)
        rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)

        # new
        # rbox = get_rbox_from_binary_mask(binary_mask, self.img_info['width'], self.img_info['height'])
        # rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)

        # new_bbox_xyxy, fix = calculate_bbox_and_fix(rbox)
        # bbox_tensor = torch.as_tensor(new_bbox_xyxy, dtype=torch.float32)#xyxy

        return bbox_tensor, rbox_tensor, mask_tensor