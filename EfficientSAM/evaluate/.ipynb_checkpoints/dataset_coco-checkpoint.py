import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json
from pycocotools import mask as maskUtils


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def to_tensor(ann):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if "bbox" in ann:
        ann["bbox"] = torch.as_tensor(ann["bbox"], dtype=torch.float32).reshape(-1, 4)
    if "new_bbox" in ann:
        ann["new_bbox"] = torch.as_tensor(ann["new_bbox"], dtype=torch.float32).reshape(-1, 4)
    if "rbox" in ann:
        ann["rbox"] = torch.as_tensor(ann["rbox"], dtype=torch.float32).reshape(-1, 4, 2)
    if "fix" in ann:
        ann["fix"] = torch.as_tensor(ann["fix"], dtype=torch.float32).reshape(-1, 4)
    if "ratio" in ann:
        ann["ratio"] = torch.as_tensor(ann["ratio"], dtype=torch.float32).reshape(-1, 1)
    if "caption" in ann:
        ann["caption"] = preprocess_caption(ann["caption"])
    return ann


class COCODataset_rbox(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, max_data_size=None):
        """
        初始化数据集类
        :param ann_file: 新生成的COCO标注文件路径
        :param data_dir: 图像数据所在目录
        :param transform: 图像预处理转换操作
        """
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform
        self.max_data_size = max_data_size

        # 加载标注文件
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        self.annotations = self.coco_data['annotations']

        # 根据 max_data_size 截取数据集
        if max_data_size is not None and len(self.annotations) > max_data_size:
            self.annotations = self.annotations[:max_data_size]
            # self.annotations = self.annotations[-max_data_size:]

    def __len__(self):
        """
        返回数据集的长度，即标注信息的数量
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        根据索引获取一条标注信息，返回处理后的图像和标注信息字典
        :param idx: 索引
        :return: image（处理后的图像）, target（标注信息字典）
        """
        ann = self.annotations[idx]
        filename = ann['filename']
        img_path = os.path.join(self.image_dir, filename)

        # 打开图像
        image = Image.open(img_path).convert('RGB')
        target = to_tensor(ann.copy())
        if self.transform:
            # 对图像进行预处理
            image, target = self.transform(image, target)

        return image, target


class COCODataset_AllAnnotations(Dataset):
    def __init__(self, ann_file, image_dir, max_data_size=None):
        """
        初始化数据集类
        :param ann_file: 新生成的COCO标注文件路径
        :param image_dir: 图像数据所在目录
        :param transform: 图像预处理转换操作
        :param max_data_size: 最大数据量
        """
        self.ann_file = ann_file
        self.image_dir = image_dir
        self.max_data_size = max_data_size

        # 加载标注文件
        with open(ann_file, 'r') as f:
            self.coco_data = json.load(f)

        # 为了方便根据image_id查找图片信息，将图片信息存储在字典中
        self.image_id_to_name = {img['id']: (img['file_name'], img['width'], img['height']) for img in
                                 self.coco_data['images']}

        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.image_annotations:
                self.image_annotations[image_id] = []
            self.image_annotations[image_id].append(ann)

        self.image_ids = list(self.image_annotations.keys())

        # 根据 max_data_size 截取数据集
        if max_data_size is not None and len(self.image_ids) > max_data_size:
            # self.image_ids = self.image_ids[:max_data_size]
            self.image_ids = self.image_ids[-max_data_size:]

    def __len__(self):
        """
        返回数据集的长度，即图像的数量
        """
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        根据索引获取一张图像的所有标注信息，返回处理后的图像和标注信息列表
        :param idx: 索引
        :return: image（处理后的图像）, targets（标注信息列表）
        """
        image_id = self.image_ids[idx]
        annotations = self.image_annotations[image_id]

        # origin coco
        filename, w, h = self.image_id_to_name[annotations[0]['image_id']]
        # new coco
        # first_ann = annotations[0]
        # filename = first_ann['filename']

        img_path = os.path.join(self.image_dir, filename)

        return img_path, annotations, w, h


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


class COCODataset_OneImage(Dataset):
    def __init__(self, all_annotations, w, h):
        """
        初始化数据集类
        :param all_annotations: 单张图像的所有标注信息列表
        """
        self.all_annotations = all_annotations
        self.w = w
        self.h = h

    def __len__(self):
        """
        返回数据集的长度，即标注信息的数量
        """
        return len(self.all_annotations)

    def __getitem__(self, idx):
        """
        根据索引获取一条标注信息的 segmentation 和 bbox
        :param idx: 索引
        :return: segmentation（分割信息）, bbox（边界框信息）
        """
        ann = self.all_annotations[idx]

        if ann['iscrowd'] == 0:
            polygons = ann['segmentation']
            mask = np.zeros((self.h, self.w), dtype=np.uint8)
            for polygon in polygons:
                # 将每个 torch.Tensor 对象转换为 Python 浮点数
                polygon_list = [tensor.item() for tensor in polygon]
                points = np.array(polygon_list).reshape(-1, 2).astype(np.int32)
                # 使用 fillPoly 函数填充多边形区域
                cv2.fillPoly(mask, [points], 1)
            mask_tensor = torch.tensor(mask, dtype=torch.float32)  # h * w
        else:
            rle = ann['segmentation']
            rle["counts"] = torch.cat(rle["counts"]).numpy()
            # 创建RLE对象
            rle_obj = maskUtils.frPyObjects(rle, rle["size"][0], rle["size"][1])
            # 解码RLE对象为掩码
            binary_mask = maskUtils.decode(rle_obj)
            # 直接将 binary_mask 转换为 torch.Tensor
            mask_tensor = torch.tensor(binary_mask, dtype=torch.float32)  # h * w

        # origin rbox
        # rbox = torch.as_tensor(ann["rbox"], dtype=torch.float32).reshape(4, 2)

        # new rbox
        rbox = get_rbox_from_bbox_with_fix(ann["bbox"], ann['fix'])
        rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)
        bbox_tensor = torch.as_tensor(ann["bbox"], dtype=torch.float32)

        # dino coco rbox
        # rbox=get_rbox_from_bbox_with_fix(ann["dino_bbox"],ann['dino_fix'])
        # rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)
        # bbox_tensor = torch.as_tensor(ann["dino_bbox"], dtype=torch.float32)

        # return rbox, mask_tensor
        return bbox_tensor, rbox_tensor, mask_tensor


class COCODataset_OneImage_bbox(Dataset):
    def __init__(self, all_annotations):
        """
        初始化数据集类
        :param all_annotations: 单张图像的所有标注信息列表
        """
        self.all_annotations = all_annotations

    def __len__(self):
        """
        返回数据集的长度，即标注信息的数量
        """
        return len(self.all_annotations)

    def __getitem__(self, idx):
        """
        根据索引获取一条标注信息的 segmentation 和 bbox
        :param idx: 索引
        :return: segmentation（分割信息）, bbox（边界框信息）
        """
        ann = self.all_annotations[idx]

        if ann['iscrowd'] == 0:
            polygons = ann['segmentation']
            mask = np.zeros((ann['img_height'], ann['img_width']), dtype=np.uint8)
            for polygon in polygons:
                # 将每个 torch.Tensor 对象转换为 Python 浮点数
                polygon_list = [tensor.item() for tensor in polygon]
                points = np.array(polygon_list).reshape(-1, 2).astype(np.int32)
                # 使用 fillPoly 函数填充多边形区域
                cv2.fillPoly(mask, [points], 1)
            mask_tensor = torch.tensor(mask, dtype=torch.float32)  # h * w
        else:
            rle = ann['segmentation']
            rle["counts"] = torch.cat(rle["counts"]).numpy()
            # 创建RLE对象
            rle_obj = maskUtils.frPyObjects(rle, rle["size"][0], rle["size"][1])
            # 解码RLE对象为掩码
            binary_mask = maskUtils.decode(rle_obj)
            # 直接将 binary_mask 转换为 torch.Tensor
            mask_tensor = torch.tensor(binary_mask, dtype=torch.float32)  # h * w

        bbox = torch.as_tensor(ann["bbox"], dtype=torch.float32)

        return bbox, mask_tensor




