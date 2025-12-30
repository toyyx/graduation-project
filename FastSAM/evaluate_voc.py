import json
import logging
import os

import cv2
import numpy as np
import torch
from pycocotools import mask as maskUtils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from lvis import LVIS
from fastsam import FastSAM, FastSAMPrompt



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

def calculate_ratio(bbox,rbox):
    """
    计算最小水平外接框、偏移比值和面积比值
    :param rbox: 旋转包围框的四个点坐标
    :return: new_box, fix, ratio
    """

    rbox = np.array(rbox)
    bbox_width = bbox[2]
    bbox_height = bbox[3]

    # 计算 rbox 与 bbox 面积的比值
    rbox_area = cv2.contourArea(rbox.astype(np.int32))
    bbox_area = bbox_width * bbox_height
    ratio = rbox_area / bbox_area if bbox_area > 0 else 0

    return ratio

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
    average_iou = sum(batch_ious) / len(batch_ious) if batch_ious else 0
    return average_iou

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
        ratio = calculate_ratio(bbox, rbox)
        ratio_tensor = torch.as_tensor(ratio, dtype=torch.float32)  # 4

        return bbox_tensor, rbox_tensor,ratio_tensor, mask_tensor

def process_single_image(model, image_file_path):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    everything_results = model(image_file_path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, )
    prompt_process = FastSAMPrompt(image_file_path, everything_results, device=DEVICE)

    return prompt_process

def process_single_mask(prompt_process, bbox_tensor, rbox_tensor, ratio_tensor):
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
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    bbox_tensor = xywh_to_xyxy(bbox_tensor)

    #masks = prompt_process.box_prompt(bboxes=bbox_tensor.cpu().numpy())
    #masks = prompt_process.rbox_prompt(rboxes=rbox_tensor.cpu().numpy())
    masks = prompt_process.bbox_rbox_prompt(bboxes=bbox_tensor.cpu().numpy(),rboxes=rbox_tensor.cpu().numpy(),ratio_tensor=ratio_tensor)

    return torch.as_tensor(masks, device='cpu')
    #return masks

model = FastSAM('./weights/FastSAM-x.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(filename='eva_voc_rbox.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

dataset_folder_path = r'C:\Users\yangxinyao\Downloads\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'
image_name_txt_path=os.path.join(dataset_folder_path,'ImageSets','Segmentation','trainval.txt')
image_dir =os.path.join(dataset_folder_path,'JPEGImages')
seg_image_dir =os.path.join(dataset_folder_path,'SegmentationObject')
annotation_dir = os.path.join(dataset_folder_path,'Annotations')

num_val = None
path_dataset = PascalVOCImagePathDataset(image_name_txt_path,image_dir,seg_image_dir,annotation_dir,num_val)
path_dataloader = DataLoader(path_dataset, batch_size=1)
num_val=len(path_dataloader)

total_iou = 0
total_num = 0
with torch.no_grad():
    # 遍历文件夹中的所有文件
    for image_index, (image_path, instance_seg_path, annotation_path) in tqdm(enumerate(path_dataloader, start=1), total=len(path_dataloader),desc=f'Evaluate ==> '):
        dataset = PascalVOCInstanceSegDataset(instance_seg_path[0])
        dataloader = DataLoader(dataset, batch_size=4)
        total_num += len(dataloader)
        prompt_process = None
        for batch_idx, (bbox_tensor, rbox_tensor,ratio_tensor, gt_mask_tensor) in enumerate(dataloader,start=1):  # point_tensor:b12    gt_mask_tensor:bhw
            if prompt_process is None:
                prompt_process = process_single_image(model, image_path[0])  # bchw
            masks = process_single_mask(prompt_process, bbox_tensor, rbox_tensor, ratio_tensor)  # bchw

            iou = calculate_iou(masks, gt_mask_tensor)
            total_iou += iou
            print(f'VAL ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, iou:{iou}')
            logging.info(f'VAL ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, iou:{iou}')
avg_iou = total_iou / total_num
print(f'VAL ==> IoU: {avg_iou}')
logging.info(f'VAL ==> IoU: {avg_iou}')