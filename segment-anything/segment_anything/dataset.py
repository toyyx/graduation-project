import os
import json
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
import matplotlib
from torch.utils.data import Dataset

matplotlib.use('TkAgg')  # 设置后端为 Agg
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import cv2

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

def get_new_bbox_from_binary_mask(binary_mask, img_width, img_height):
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
    #sorted_box = sort_rbox_points(box)
    
    rbox = np.array(box)
    x_min = np.min(rbox[:, 0])
    y_min = np.min(rbox[:, 1])
    x_max = np.max(rbox[:, 0])
    y_max = np.max(rbox[:, 1])
   
    new_bbox_xyxy = [x_min, y_min, x_max, y_max]
    return new_bbox_xyxy

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

def xywh_to_xyxy(boxes):
    """
    将 xywh 格式的边界框转换为 xyxy 格式
    :param boxes: 形状为 (N, 4) 的张量，N 是边界框数量，每个边界框是 [x, y, w, h] 格式
    :return: 形状为 (N, 4) 的张量，每个边界框是 [x1, y1, x2, y2] 格式
    """
    x = boxes[0]
    y = boxes[1]
    w = boxes[2]
    h = boxes[3]
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    return [x1, y1, x2, y2]

def expand_boxes_xyxy(boxes, image_width, image_height, expand):
    """
    此函数用于将边界框扩大 110% 并调整其坐标不超出图片范围
    :param boxes: 形状为 (N, 4) 的张量，N 是边界框数量，每个边界框是 [x1, y1, x2, y2] 格式
    :param image_width: 图片的宽度
    :param image_height: 图片的高度
    :return: 扩大并调整后的边界框坐标
    """
        
    # 计算边界框的宽度和高度
    widths = boxes[2] - boxes[0]
    heights = boxes[3] - boxes[1]
    # 计算边界框的中心点坐标
    cxs = boxes[0] + widths / 2
    cys = boxes[1] + heights / 2

    # 扩大宽度和高度到原来的 110%
    new_widths = widths * expand
    new_heights = heights * expand

    # 计算扩大后的边界框坐标
    x1_new = cxs - new_widths / 2
    y1_new = cys - new_heights / 2
    x2_new = cxs + new_widths / 2
    y2_new = cys + new_heights / 2

    # 调整边界框坐标不超出图片范围
    x1_new = np.clip(x1_new, 0, image_width)
    y1_new = np.clip(y1_new, 0, image_height)
    x2_new = np.clip(x2_new, 0, image_width)
    y2_new = np.clip(y2_new, 0, image_height)

    # 组合成新的边界框
    new_boxes = [x1_new, y1_new, x2_new, y2_new]
    return new_boxes
    
def get_rbox_from_new_bbox(new_bbox,fix):
    x1 = new_bbox[0]
    y1 = new_bbox[1]
    x2 = new_bbox[2]
    y2 = new_bbox[3]

    # 计算边界框的宽度和高度
    widths = x2 - x1
    heights = y2 - y1
    top_point=[x1 + widths * fix[0], y1]
    right_point=[x2 , y1 + heights * fix[1]]
    bottom_point=[x2 - widths * fix[2], y2]
    left_point=[x1 , y2 - heights * fix[3]]

    return [top_point,right_point,bottom_point,left_point]

# 定义处理单个 JSON 文件的数据集类
class SingleJSONDataset(Dataset):
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        self.image_info = json_data['image']
        self.annotations = json_data['annotations']
        self.count=len(self.annotations)
        self.now = 0

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        self.now += 1
        annotation = self.annotations[idx]

        point_coords = annotation['point_coords']
        point_tensor = torch.tensor(point_coords, dtype=torch.float32)# points_number * 2

        segmentation = annotation['segmentation']
        binary_mask = mask_utils.decode(segmentation) # 解码COCO RLE格式的分割掩码
        # 直接将 binary_mask 转换为 torch.Tensor
        mask_tensor = torch.tensor(binary_mask, dtype=torch.float32)# h * w

        return point_tensor, mask_tensor
    
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
        #point_coords = annotation['point_coords']
        #point_tensor = torch.tensor(point_coords, dtype=torch.float32)# points_number * 2
        
        segmentation = annotation['segmentation']
        binary_mask = mask_utils.decode(segmentation) # 解码COCO RLE格式的分割掩码
        mask_tensor = torch.tensor(binary_mask, dtype=torch.float32)# h * w
        
        #bbox_tensor = torch.as_tensor(annotation["bbox"], dtype=torch.float32)# 4
        
        #origin rbox
        #rbox = get_rbox_from_binary_mask(binary_mask,self.image_info['width'],self.image_info['height'])
        #rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)
        
        #new_bbox_xyxy,fix=calculate_bbox_and_fix(rbox)
        #bbox_tensor = torch.as_tensor(new_bbox_xyxy, dtype=torch.float32)# 4
        
        #new rbox
        rbox = get_rbox_from_binary_mask(binary_mask,self.image_info['width'],self.image_info['height'])
        new_bbox_xyxy,fix=calculate_bbox_and_fix(rbox)
        rbox=get_rbox_from_bbox_with_fix(annotation["bbox"],fix)
        rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)
        bbox_tensor = torch.as_tensor(annotation["bbox"], dtype=torch.float32)# 4
        
        
        #expand rbox
        #rbox = get_rbox_from_binary_mask(binary_mask,self.image_info['width'],self.image_info['height'])
        #new_bbox_xyxy,fix=calculate_bbox_and_fix(rbox)
        #expand_new_bbox = expand_boxes_xyxy(new_bbox_xyxy,self.image_info['width'],self.image_info['height'],expand=1.1)
        #rbox = get_rbox_from_bbox(expand_new_bbox,fix)
        #rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)
        
        #return rbox_tensor, mask_tensor
        return bbox_tensor, rbox_tensor, mask_tensor
    
class SingleJSONDataset_with_bbox(Dataset):
    def __init__(self, json_file_path):
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        self.image_info = json_data['image']
        self.annotations = json_data['annotations']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        #point_coords = annotation['point_coords']
        #point_tensor = torch.tensor(point_coords, dtype=torch.float32)# points_number * 2
        
        segmentation = annotation['segmentation']
        binary_mask = mask_utils.decode(segmentation) # 解码COCO RLE格式的分割掩码
        mask_tensor = torch.tensor(binary_mask, dtype=torch.float32)# h * w
        
        #origin bbox
        #bbox_tensor = torch.as_tensor(annotation["bbox"], dtype=torch.float32)# 4
        
        #new bbox
        rbox = get_rbox_from_binary_mask(binary_mask,self.image_info['width'],self.image_info['height'])
        new_bbox_xyxy,fix=calculate_bbox_and_fix(rbox)
        bbox_tensor=torch.as_tensor(new_bbox_xyxy, dtype=torch.float32)
        #rbox = get_rbox_from_binary_mask(binary_mask,self.image_info['width'],self.image_info['height'])
        #rbox_tensor = torch.as_tensor(rbox, dtype=torch.float32).reshape(4, 2)

        return bbox_tensor, mask_tensor


def read_data_from_folder(folder_path):
    data = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            # 获取图片文件名和对应的JSON文件名
            image_filename = filename
            json_filename = os.path.splitext(image_filename)[0] + '.json'
            json_file_path = os.path.join(folder_path, json_filename)
            image_file_path = os.path.join(folder_path, image_filename)

            # 检查对应的JSON文件是否存在
            if os.path.exists(json_file_path):
                # 读取JSON文件
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)

                # 提取图片信息
                image_info = json_data['image']
                #image_id = image_info['image_id']
                #width = image_info['width']
                #height = image_info['height']
                file_name = image_info['file_name']

                # 读取图片
                #image = Image.open(image_file_path)

                # 提取标注信息
                annotations = json_data['annotations']
                print(f'file:{file_name} annotations:{len(annotations)}')

                
                for annotation in annotations:
                    annotation_id = annotation['id']
                    segmentation = annotation['segmentation']
                    bbox = annotation['bbox']
                    area = annotation['area']
                    predicted_iou = annotation['predicted_iou']
                    stability_score = annotation['stability_score']
                    crop_box = annotation['crop_box']
                    point_coords = annotation['point_coords']
                    print(f'point_coords:{len(point_coords)}')
                    # 解码COCO RLE格式的分割掩码
                    #binary_mask = mask_utils.decode(segmentation)


                    """
                    
                    
                    # 存储数据
                    data_item = {
                        'image_id': image_id,
                        'width': width,
                        'height': height,
                        'file_name': file_name,
                        'image': image,
                        'annotation_id': annotation_id,
                        'segmentation': segmentation,
                        'binary_mask': binary_mask,
                        'bbox': bbox,
                        'area': area,
                        'predicted_iou': predicted_iou,
                        'stability_score': stability_score,
                        'crop_box': crop_box,
                        'point_coords': point_coords
                    }
                    data.append(data_item)
                    """
                    


    return data

def show_mask_on_image(data):
    for item in data:
        image = item['image']
        binary_mask = item['binary_mask']

        # 将PIL图像转换为numpy数组
        image_np = np.array(image)

        # 创建一个颜色掩码，将掩码区域设置为特定颜色（这里使用红色）
        color_mask = np.zeros_like(image_np)
        color_mask[binary_mask > 0] = [255, 0, 0]  # 红色

        # 叠加掩码到原始图像上
        alpha = 0.5  # 透明度
        overlay = image_np.copy()
        overlay = np.where(binary_mask[..., None] > 0,
                           (1 - alpha) * overlay + alpha * color_mask,
                           overlay).astype(np.uint8)

        # 显示图像和掩码
        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(f"Image ID: {item['image_id']}, Annotation ID: {item['annotation_id']}")
        plt.axis('off')
        plt.show()

# 指定文件夹路径
#folder_path = 'E:\\An_-m2SWozW4o-FatJEIY1Anj32x8TnUqad9WMAVkMaZHkDyHfjpLcVlQoTFhgQihg8U4R5KqJvoJrtBwT3eKH-Yj5-LfY0'
#folder_path = 'C:\\Users\\yangxinyao\\Downloads\\testdata'
#data = read_data_from_folder(folder_path)

# 打印读取的数据数量
#print(f"Read {len(data)} data items from the folder.")

# 展示掩码在图像上
#show_mask_on_image(data)


