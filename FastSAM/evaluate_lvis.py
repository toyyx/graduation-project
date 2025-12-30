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

    # 计算 rbox 与 bbox 面积的比值
    rbox_area = cv2.contourArea(rbox.astype(np.int32))
    bbox_area = bbox_width * bbox_height
    ratio = rbox_area / bbox_area if bbox_area > 0 else 0

    return new_bbox_xyxy, fix, ratio

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

class LVISFullDataset(Dataset):
    def __init__(self, lvis_api, image_root_path ,max=None):
        """
        初始化 LVISFullDataset 类
        :param json_path: LVIS 标注文件的 JSON 路径
        :param image_root_path: 图片的根路径
        """
        self.lvis_api = lvis_api
        self.img_ids = self.lvis_api.get_img_ids()
        self.image_root_path = image_root_path

        if max is not None and max <= len(self.img_ids):
            self.img_ids =self.img_ids[:max]


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
        #bbox = np.array(ann['bbox'], dtype=np.float32)
        bbox_tensor = torch.as_tensor(ann["bbox"], dtype=torch.float32)  # xywh

        rbox = get_rbox_from_binary_mask(binary_mask, self.img_info['width'], self.img_info['height'])
        _, fix ,_= calculate_bbox_and_fix(rbox)
        rbox = get_rbox_from_bbox_with_fix(ann["bbox"] ,fix)
        rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)
        ratio = calculate_ratio(ann["bbox"], rbox)
        ratio_tensor = torch.as_tensor(ratio, dtype=torch.float32)  # 4



        # new
        # rbox = get_rbox_from_binary_mask(binary_mask, self.img_info['width'], self.img_info['height'])
        # rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)

        # new_bbox_xyxy, fix = calculate_bbox_and_fix(rbox)
        # bbox_tensor = torch.as_tensor(new_bbox_xyxy, dtype=torch.float32)#xyxy

        return bbox_tensor, rbox_tensor, ratio_tensor,mask_tensor

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

    #masks = prompt_process.rbox_prompt(rboxes=rbox_tensor.cpu().numpy())

    masks = prompt_process.bbox_rbox_prompt(bboxes=bbox_tensor.cpu().numpy(),rboxes=rbox_tensor.cpu().numpy(),ratio_tensor=ratio_tensor)
    #bbox_tensor = xywh_to_xyxy(bbox_tensor)
    #masks = prompt_process.box_prompt(bboxes=bbox_tensor.cpu().numpy())
    return torch.as_tensor(masks, device='cpu')
    #return masks

# 自定义 collate_fn 函数，直接返回原始数据
def custom_collate(batch):
    #print(batch)
    data=batch[0]
    return data[0],data[1],data[2]

model = FastSAM('./weights/FastSAM-x.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(filename='eva_lvis_rbox.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

lvis_json_path = r'C:\Users\yangxinyao\Downloads\lvis_v1_val.json\lvis_v1_val.json'
coco_image_root_path = r'C:\Users\yangxinyao\Downloads\coco2017'

lvis_api = LVIS(annotation_path=lvis_json_path)

num_val = None
image_dataset = LVISFullDataset(lvis_api, coco_image_root_path, max=num_val)
image_dataloader = DataLoader(image_dataset, batch_size=1, collate_fn=custom_collate)
num_val = len(image_dataloader)

total_iou = 0
total_num = 0
with torch.no_grad():
    # 遍历文件夹中的所有文件 full_img_path, img_info, anns
    for image_index, (image_path, img_info, anns) in tqdm(enumerate(image_dataloader, start=1),total=len(image_dataloader),desc=f'Evaluate ==> '):
        dataset = LVISAnnotationDataset(lvis_api,img_info, anns)
        dataloader = DataLoader(dataset, batch_size=8)
        total_num += len(dataloader)
        prompt_process = None
        for batch_idx, (bbox_tensor, rbox_tensor,ratio_tensor, gt_mask_tensor) in enumerate(dataloader, start=1):  # point_tensor:b12    gt_mask_tensor:bhw
            if prompt_process is None:
                prompt_process = process_single_image(model, image_path)  # bchw
            masks = process_single_mask(prompt_process, bbox_tensor, rbox_tensor, ratio_tensor) # bchw

            iou = calculate_iou(masks, gt_mask_tensor)
            total_iou += iou
            print(f'VAL ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, iou:{iou}')
            logging.info(f'VAL ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, iou:{iou}')
avg_iou = total_iou / total_num
print(f'VAL ==> IoU: {avg_iou}')
logging.info(f'VAL ==> IoU: {avg_iou}')