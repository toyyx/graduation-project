import os
import cv2
import numpy as np
import json
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from tqdm import tqdm

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

def get_rbox_from_polygons(polygons, img_width, img_height):
    """
    从多边形列表获取旋转包围框
    :param polygons: 多边形列表，每个元素是一个多边形的坐标列表
    :return: 旋转包围框的四个点坐标
    """
    all_points = []
    for polygon in polygons:
        points = np.array(polygon).reshape(-1, 2).astype(np.int32)
        all_points.extend(points)
    all_points = np.array(all_points)
    rect = cv2.minAreaRect(all_points)
    box = cv2.boxPoints(rect)
    # 裁剪超出图片范围的点
    box[:, 0] = np.clip(box[:, 0], 0, img_width)
    box[:, 1] = np.clip(box[:, 1], 0, img_height)
    # 排序 rbox 的点
    sorted_box = sort_rbox_points(box)
    return sorted_box


def get_rbox_from_rle(rle, img_width, img_height):
    """
    从RLE编码获取旋转包围框
    :param rle: RLE编码
    :return: 旋转包围框的四个点坐标
    """
    # 创建RLE对象
    rle_obj = maskUtils.frPyObjects(rle, rle["size"][0], rle["size"][1])
    # 解码RLE对象为掩码
    mask = maskUtils.decode(rle_obj)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    new_bbox = [x_min, y_min, bbox_width, bbox_height]

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

    return new_bbox, fix, ratio


# 数据集路径
dataDir = r'C:\Users\yangxinyao\Downloads\coco2017'
dataType = 'train2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

try:
    # 初始化COCO api
    coco = COCO(annFile)
except FileNotFoundError:
    print(f"未找到注释文件，请检查路径 {annFile} 是否正确。")
    exit(1)
except Exception as e:
    print(f"初始化COCO API时出现错误: {e}")
    exit(1)

# 获取所有图像的ID
imgIds = coco.getImgIds()

# 加载原始的COCO数据
with open(annFile, 'r') as f:
    coco_data = json.load(f)

# 存储补充信息后的标注
new_annotations = []

for img_id in tqdm(imgIds, desc="Processing images"):
    img = coco.loadImgs(img_id)[0]
    img_width = img['width']
    img_height = img['height']
    # 获取该图像的注释ID
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=coco.getCatIds(), iscrowd=None)
    anns = coco.loadAnns(annIds)

    for ann in tqdm(anns, desc=f"Processing annotations for image {img_id}", leave=False):
        if ann['iscrowd'] == 0:
            polygons = ann['segmentation']
            assert isinstance(polygons, list) and all(isinstance(poly, list) for poly in polygons), f"{img_id}--{ann}提供的 segmentation 不是有效的多边形格式。"
            rbox = get_rbox_from_polygons(polygons, img_width, img_height)
        else:
            rle = ann['segmentation']
            assert isinstance(rle, dict) and 'counts' in rle and 'size' in rle, f"{img_id}--{ann}提供的 segmentation 不是有效的 RLE 格式。"
            rbox = get_rbox_from_rle(rle, img_width, img_height)
        ann['rbox'] = rbox

        new_bbox, fix, ratio = calculate_bbox_and_fix(rbox)
        ann['new_bbox'] = new_bbox #xywh
        ann['fix'] = fix
        ann['ratio'] = ratio

        ann['filename'] = img['file_name']
        ann['img_width'] = img_width
        ann['img_height'] = img_height

        new_annotations.append(ann)

# 更新COCO数据中的标注信息
coco_data['annotations'] = new_annotations

# 保存为新的JSON文件
output_file = '{}/annotations/instances_{}_with_rbox.json'.format(dataDir, dataType)
with open(output_file, 'w') as f:
    json.dump(coco_data, f)

print(f"补充信息后的标注已保存到 {output_file}")


