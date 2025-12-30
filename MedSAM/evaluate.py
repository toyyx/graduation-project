import json

import matplotlib
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw


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

# 定义处理单个 JSON 文件的数据集类
class SingleJSONDataset_with_rbox(Dataset):

    def __init__(self, json_file_path):
    #def __init__(self, json_data):
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        self.image_info = json_data['image']
        self.annotations = json_data['annotations']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        annotation = self.annotations[idx]
        # point_coords = annotation['point_coords']
        # point_tensor = torch.tensor(point_coords, dtype=torch.float32)# points_number * 2

        segmentation = annotation['segmentation']
        binary_mask = mask_utils.decode(segmentation)  # 解码COCO RLE格式的分割掩码
        mask_tensor = torch.tensor(binary_mask, dtype=torch.float32)  # h * w

        # bbox_tensor = torch.as_tensor(annotation["bbox"], dtype=torch.float32)# 4

        # origin rbox
        # rbox = get_rbox_from_binary_mask(binary_mask,self.image_info['width'],self.image_info['height'])
        # rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)

        # new_bbox_xyxy,fix=calculate_bbox_and_fix(rbox)
        # bbox_tensor = torch.as_tensor(new_bbox_xyxy, dtype=torch.float32)# 4

        # new rbox
        rbox = get_rbox_from_binary_mask(binary_mask, self.image_info['width'], self.image_info['height'])
        _, fix, _ = calculate_bbox_and_fix(rbox)
        rbox = get_rbox_from_bbox_with_fix(annotation["bbox"], fix)
        rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32,device=device).reshape(4, 2)
        bbox_tensor = torch.as_tensor(annotation["bbox"], dtype=torch.float32,device=device)  # 4

        # expand rbox
        # rbox = get_rbox_from_binary_mask(binary_mask,self.image_info['width'],self.image_info['height'])
        # new_bbox_xyxy,fix=calculate_bbox_and_fix(rbox)
        # expand_new_bbox = expand_boxes_xyxy(new_bbox_xyxy,self.image_info['width'],self.image_info['height'],expand=1.1)
        # rbox = get_rbox_from_bbox(expand_new_bbox,fix)
        # rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)

        # return rbox_tensor, mask_tensor
        return bbox_tensor, rbox_tensor, mask_tensor

class MedSAM(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=box_torch,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks

def init_model(checkpoint_path=None):
    model_type = "vit_b"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    sam_model = sam_model_registry[model_type]()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    log = sam_model.load_state_dict(checkpoint, strict=False)
    print("Model loaded from {} \n => {}".format(checkpoint_path, log))

    rbox_path='sam_conbine_epoch_78_5lr5_rbox_best.pth'
    checkpoint2 = torch.load(rbox_path, map_location=device)
    sam_model.prompt_encoder.rbox_point_embeddings[0].weight.data = checkpoint2['prompt_encoder.rbox_point_embeddings.0.weight']
    sam_model.prompt_encoder.rbox_point_embeddings[1].weight.data = checkpoint2['prompt_encoder.rbox_point_embeddings.1.weight']
    sam_model.prompt_encoder.rbox_point_embeddings[2].weight.data = checkpoint2['prompt_encoder.rbox_point_embeddings.2.weight']
    sam_model.prompt_encoder.rbox_point_embeddings[3].weight.data = checkpoint2['prompt_encoder.rbox_point_embeddings.3.weight']

    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.eval()

    return medsam_model

def process_single_image(medsam_model,image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    img_np = io.imread(image_path)
    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape

    # % image preprocessing
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
    return image_embedding,H, W

def process_single_mask(medsam_model,img_embed,bbox_tensor,rbox_tensor,H,W):
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

    bbox_tensor=xywh_to_xyxy(bbox_tensor)
    scale_tensor = torch.tensor([W, H, W, H], dtype=torch.float, device=bbox_tensor.device).unsqueeze(0)
    boxes_1024_tensor = bbox_tensor / scale_tensor * 1024
    scale_tensor_rbox = torch.tensor([W, H], dtype=torch.float, device=rbox_tensor.device).unsqueeze(0).unsqueeze(1)
    rbox_1024_tensor = rbox_tensor / scale_tensor_rbox * 1024

    if len(boxes_1024_tensor.shape) == 2:
        boxes_1024_tensor = boxes_1024_tensor[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=boxes_1024_tensor,
        #rboxes=None,
        rboxes=rbox_1024_tensor,
        masks=None,
    )
    low_res_masks, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    #low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    ori_res_masks = F.interpolate(
        low_res_masks,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    #low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    #medsam_seg = (low_res_pred > 0.5).astype(np.uint8) #b
    return ori_res_masks


checkpoint_path = r'./medsam_vit_b.pth'
medsam = init_model(checkpoint_path)


logging.basicConfig(filename='eva_sa1b_rbox.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

dataset_folder_path = r'C:\Users\yangxinyao\Downloads\1100sa1b'
all_image_files = [f for f in os.listdir(dataset_folder_path) if f.endswith('.jpg')]
num_val = 100

# 划分数据集
val_image_files = all_image_files[-num_val:]

# 验证阶段
total_iou = 0
total_num = 0
with torch.no_grad():
    # 遍历文件夹中的所有文件
    for image_index, image_filename in tqdm(enumerate(val_image_files, start=1), total=len(val_image_files),desc=f'Val ==> '):
        json_filename = os.path.splitext(image_filename)[0] + '.json'
        json_file_path = os.path.join(dataset_folder_path, json_filename)
        image_file_path = os.path.join(dataset_folder_path, image_filename)
        image_embeddings = None

        # 检查对应的JSON文件是否存在
        if os.path.exists(json_file_path):
            dataset = SingleJSONDataset_with_rbox(json_file_path)
            dataloader = DataLoader(dataset, batch_size=16)
            total_num += len(dataloader)
            for batch_idx, (bbox_tensor, rbox_tensor, gt_mask_tensor) in enumerate(dataloader,start=1):
                if image_embeddings is None:
                    image_embeddings, H, W = process_single_image(medsam, image_file_path)
                masks = process_single_mask(medsam, image_embeddings,  bbox_tensor,rbox_tensor,H, W).cpu()

                iou = calculate_iou(masks.squeeze(1) > 0.0, gt_mask_tensor)
                total_iou += iou
                print(f'VAL ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, iou:{iou}')
                logging.info(f'VAL ==> file:{image_index}/{num_val}  batch:{batch_idx}/{len(dataloader)}, iou:{iou}')

                """
                # 可视化部分
                image = Image.open(image_file_path)
                draw = ImageDraw.Draw(image)

                # 绘制边界框
                for bbox in bbox_tensor:
                    x, y, w, h = bbox
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

                # 绘制旋转框
                for rbox in rbox_tensor:
                    points = rbox.tolist()
                    # 将每个点的坐标转换为 float 类型
                    points = [(point[0],point[1]) for point in points]
                    # 确保点是封闭的，即第一个点和最后一个点相同
                    points.append(points[0])
                    draw.line(points, fill='yellow', width=2)

                # 绘制掩码
                alpha = 0.4  # 掩码透明度
                colored_mask_masks = np.zeros((H, W, 4), dtype=np.uint8)  # 用于 masks 的掩码
                colored_mask_gt = np.zeros((H, W, 4), dtype=np.uint8)  # 用于 gt_mask_tensor 的掩码

                # 处理 masks
                for single_mask in masks.squeeze(1) > 0.0:
                    single_mask = single_mask.numpy().astype(np.uint8)
                    single_mask = (single_mask * 255).astype(np.uint8)
                    single_mask = Image.fromarray(single_mask, mode='L').resize((W, H))
                    single_mask = np.array(single_mask) > 0
                    colored_mask_masks[single_mask, 1] = 255  # 绿色通道，代表 masks 为绿色
                    colored_mask_masks[single_mask, 3] = np.minimum(255, colored_mask_masks[single_mask, 3] + int(
                        alpha * 255))

                # 处理 gt_mask_tensor
                for single_mask in gt_mask_tensor:
                    single_mask = single_mask.numpy().astype(np.uint8)
                    single_mask = (single_mask * 255).astype(np.uint8)
                    single_mask = Image.fromarray(single_mask, mode='L').resize((W, H))
                    single_mask = np.array(single_mask) > 0
                    colored_mask_gt[single_mask, 2] = 255  # 蓝色通道，代表 gt_mask_tensor 为蓝色
                    colored_mask_gt[single_mask, 3] = np.minimum(255,
                                                                 colored_mask_gt[single_mask, 3] + int(alpha * 255))

                colored_mask_masks = Image.fromarray(colored_mask_masks)
                colored_mask_gt = Image.fromarray(colored_mask_gt)

                # 合并两个掩码
                combined_mask = Image.alpha_composite(colored_mask_masks, colored_mask_gt)
                image = Image.alpha_composite(image.convert('RGBA'), combined_mask)

                # 显示图像
                plt.imshow(image)
                plt.title(f'Batch {batch_idx}')
                plt.axis('off')
                plt.show()
                """



avg_iou = total_iou / total_num
print(f'VAL ==> IoU: {avg_iou}')
logging.info(f'VAL ==> IoU: {avg_iou}')