import json

import matplotlib
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from fastsam import FastSAM, FastSAMPrompt
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import time

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
record_logger=init_logger("record.log")
record_logger.info(f'FastSAM')

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
        rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)
        bbox_tensor = torch.as_tensor(annotation["bbox"], dtype=torch.float32)  # 4
        ratio=calculate_ratio(annotation["bbox"],rbox)
        ratio_tensor = torch.as_tensor(ratio, dtype=torch.float32)  # 4

        # expand rbox
        # rbox = get_rbox_from_binary_mask(binary_mask,self.image_info['width'],self.image_info['height'])
        # new_bbox_xyxy,fix=calculate_bbox_and_fix(rbox)
        # expand_new_bbox = expand_boxes_xyxy(new_bbox_xyxy,self.image_info['width'],self.image_info['height'],expand=1.1)
        # rbox = get_rbox_from_bbox(expand_new_bbox,fix)
        # rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)

        # return rbox_tensor, mask_tensor
        return bbox_tensor, rbox_tensor, ratio_tensor, mask_tensor

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

def process_single(model, image_file_path, bbox_tensor, rbox_tensor):
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
    everything_results = model(image_file_path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, )
    prompt_process = FastSAMPrompt(image_file_path, everything_results, device=DEVICE)

    #masks = prompt_process.rbox_prompt(rboxes=rbox_tensor.cpu().numpy())
    bbox_tensor = xywh_to_xyxy(bbox_tensor)
    masks = prompt_process.box_prompt(bboxes=bbox_tensor.cpu().numpy())
    return torch.as_tensor(masks, device=DEVICE)

def process_single_image(model, image_file_path):
    #time1=time.time()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    everything_results = model(image_file_path, device=DEVICE, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, )
    prompt_process = FastSAMPrompt(image_file_path, everything_results, device=DEVICE)
    #time2=time.time()
    
    #speed=everything_results[0].speed
    #time=(speed['preprocess']+speed['inference']+speed['postprocess'])/1000# ms -> s
    
    mflops=everything_results[0].mflops

    #return prompt_process,time2-time1
    return prompt_process,mflops

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

    #masks = prompt_process.rbox_prompt(rboxes=rbox_tensor.cpu().numpy())
    bbox_tensor = xywh_to_xyxy(bbox_tensor)
    input_bbox=bbox_tensor.cpu().numpy()
    #input_rbox=rbox_tensor.cpu().numpy()

    
    #bbox_tensor = xywh_to_xyxy(bbox_tensor)
    
    #time1=time.time()
    masks = prompt_process.box_prompt(bboxes=input_bbox)
    #masks = prompt_process.bbox_rbox_prompt(bboxes=input_bbox,rboxes=input_rbox,ratio_tensor=ratio_tensor)
    #time2=time.time()
    
    #return torch.as_tensor(masks, device=DEVICE),time2-time1
    #return masks,time2-time1
    return masks

import cv2
import matplotlib
# 修改后端为 Agg，适合在无图形界面的环境中保存图片
matplotlib.use('Agg')
import matplotlib.pyplot as plt

model = FastSAM('./weights/FastSAM-x.pt')

"""
model_path = './weights/FastSAM-x.pt'  # 替换为你的 .pt 文件路径
try:
    state_dict = torch.load(model_path)
except FileNotFoundError:
    print(f"错误：未找到 {model_path} 文件。")
except Exception as e:
    print(f"错误：加载文件时出现问题，错误信息：{e}")
else:
    total_params = 0
    # 检查是否有 model 对象
    if 'model' in state_dict and hasattr(state_dict['model'], 'parameters'):
        model = state_dict['model']
        for param_name, param in model.named_parameters():
            # 计算当前参数的数量
            num_params = param.numel()
            total_params += num_params
            print(f"参数名: {param_name}, 参数量: {num_params}")
    else:
        # 遍历状态字典
        for param_name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                # 计算当前参数的数量
                num_params = param.numel()
                total_params += num_params
                print(f"参数名: {param_name}, 参数量: {num_params}")
            else:
                print(f"跳过非张量类型的参数: {param_name}，类型: {type(param)}")
    print(f"模型总参数量: {total_params}")

#num_params = sum(p.numel() for p in model.parameters())
#print(f"模型参数量: {num_params}")
record_logger.info(f'模型参数量: {total_params}')
"""



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


logging.basicConfig(filename='new.log', level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

dataset_folder_path = r'/root/autodl-tmp/SA-1B/val-un100'
all_image_files = [f for f in os.listdir(dataset_folder_path) if f.endswith('.jpg')]
num_val = 100

# 划分数据集
val_image_files = all_image_files[-num_val:]

# 验证阶段
total_iou = 0
total_num = 0

#total_time=0
total_mflops=0
total_sample_num=0

target_file_name=r'sa_11129'
target_batch=68
target_found=False

with torch.no_grad():
    # 遍历文件夹中的所有文件
    for image_index, image_filename in tqdm(enumerate(val_image_files, start=1), total=len(val_image_files),desc=f'Val ==> '):
        if target_found:
            break
        json_filename = os.path.splitext(image_filename)[0] + '.json'
        json_file_path = os.path.join(dataset_folder_path, json_filename)
        image_file_path = os.path.join(dataset_folder_path, image_filename)
        prompt_process=None
        
        # 从路径中提取文件名（不含扩展名）
        base_name = os.path.basename(image_file_path)
        file_name = os.path.splitext(base_name)[0]
        if file_name!=target_file_name:
            continue
        #else:
            #target_found=True

        # 检查对应的JSON文件是否存在
        if os.path.exists(json_file_path):
            dataset = SingleJSONDataset_with_rbox(json_file_path)
            # dataset = SingleJSONDataset_with_bbox(json_file_path)
            dataloader = DataLoader(dataset, batch_size=1)
            total_num += len(dataloader)
            for batch_idx, (bbox_tensor, rbox_tensor, ratio_tensor, gt_mask_tensor) in enumerate(dataloader,start=1):  # point_tensor:b12    gt_mask_tensor:bhw
                if target_found:
                    break

                if batch_idx!=target_batch:
                    continue
                else:
                    target_found=True

                if prompt_process is None:
                    prompt_process,mflops=process_single_image(model, image_file_path)# bchw
               
                masks_bbox = process_single_mask(prompt_process, bbox_tensor, rbox_tensor, ratio_tensor).cpu()  # bchw
                
                
                # 可视化部分
                image = cv2.imread(image_file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pred_mask_bbox = masks_bbox.squeeze(1)[0].numpy() > 0.0

                bbox = bbox_tensor[0].numpy().astype(int)

                fig, axes = plt.subplots(1, 1, figsize=(4, 4), frameon=False,  # 无边框
                                                             tight_layout=True)  # 紧凑布局
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)  # 移除所有边距

                # 显示预测掩码
                axes.imshow(image)
                
                rgb_color = np.array([255, 0, 0]) / 255.0
                overlay = np.zeros((*image.shape[:2], 4), dtype=float)
                overlay[pred_mask_bbox, :3] = rgb_color  
                overlay[pred_mask_bbox, 3] = 0.5  # 透明度设为0.5
                axes.imshow(overlay)
                #axes.imshow(pred_mask_bbox, alpha=0.5, cmap='jet')
                
                #axes.set_title('FastSAM')
                axes.axis('off')  # 不显示坐标轴

                print(f"Processing: {image_file_path}")

                save_dir=r'/root/autodl-tmp/target_img'
                os.makedirs(save_dir, exist_ok=True)

                # 从路径中提取文件名（不含扩展名）
                base_name = os.path.basename(image_file_path)
                file_name = os.path.splitext(base_name)[0]

                # 构建保存文件名，包含关键信息
                save_filename = f"{file_name}_batch_{batch_idx}_FastSAM.png"
                save_path = os.path.join(save_dir, save_filename)

                # 保存图像（高质量）
                plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

                print(f"Image saved to: {save_path}")

print(f'VAL ==> finish')
logging.info(f'VAL ==> finish')