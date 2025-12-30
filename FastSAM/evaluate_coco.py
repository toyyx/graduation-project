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

from fastsam import FastSAM, FastSAMPrompt

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
        rbox=get_rbox_from_bbox_with_fix(ann["bbox"],ann['fix'])
        rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)
        bbox_tensor = torch.as_tensor(ann["bbox"], dtype=torch.float32)
        ratio = calculate_ratio(ann["bbox"], rbox)
        ratio_tensor = torch.as_tensor(ratio, dtype=torch.float32)  # 4
        # dino coco rbox
        #rbox = get_rbox_from_bbox_with_fix(ann["dino_bbox"], ann['dino_fix'])
        #rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)
        #bbox_tensor = torch.as_tensor(ann["dino_bbox"], dtype=torch.float32)

        # return rbox, mask_tensor
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

    #masks = prompt_process.rbox_prompt(rboxes=rbox_tensor.cpu().numpy())

    masks = prompt_process.bbox_rbox_prompt(bboxes=bbox_tensor.cpu().numpy(),rboxes=rbox_tensor.cpu().numpy(),ratio_tensor=ratio_tensor)
    #bbox_tensor = xywh_to_xyxy(bbox_tensor)
    #masks = prompt_process.box_prompt(bboxes=bbox_tensor.cpu().numpy())
    return torch.as_tensor(masks, device='cpu')
    #return masks

model = FastSAM('./weights/FastSAM-x.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

logging.basicConfig(filename='eva_coco_0.95.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
num_val = None

val_ann_file = r'C:\Users\yangxinyao\Downloads\coco2017\annotations\instances_val2017_with_rbox.json'
val_image_dir = r'C:\Users\yangxinyao\Downloads\coco2017\val2017'

val_image_dataset = COCODataset_AllAnnotations(val_ann_file, val_image_dir, max_data_size=num_val)
val_image_dataloader = DataLoader(val_image_dataset, batch_size=1)

num_val = len(val_image_dataloader)

#model.eval()
total_iou = 0
total_num = 0
with torch.no_grad():
    for image_index, (img_path, annotations,w,h) in tqdm(enumerate(val_image_dataloader, start=1), total=len(val_image_dataloader), desc=f'Evaluate ==> '):
        oneImage_dataset = COCODataset_OneImage(annotations,w,h)
        oneImage_dataloader = DataLoader(oneImage_dataset, batch_size=4)
        total_num += len(oneImage_dataloader)
        prompt_process = None
        for batch_idx, (bbox_tensor, rbox_tensor,ratio_tensor, gt_mask_tensor) in enumerate(oneImage_dataloader,start=1):  # point_tensor:b12    gt_mask_tensor:bhw
            if prompt_process is None:
                prompt_process = process_single_image(model, img_path[0])  # bchw
            masks = process_single_mask(prompt_process, bbox_tensor, rbox_tensor, ratio_tensor) # bchw

            iou = calculate_iou(masks, gt_mask_tensor)
            total_iou += iou
            print(f'VAL ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(oneImage_dataloader)}, iou:{iou}')
            logging.info(f'VAL ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(oneImage_dataloader)}, iou:{iou}')
avg_iou = total_iou / total_num
print(f'VAL ==> IoU: {avg_iou}')
logging.info(f'VAL ==> IoU: {avg_iou}')