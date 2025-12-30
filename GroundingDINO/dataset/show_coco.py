import os
import cv2
import numpy as np
import json
import matplotlib
# 修改后端为 Agg，适合在无图形界面的环境中保存图片
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# 数据集路径
#dataDir = r'C:\Users\yangxinyao\Downloads\coco2017'
#dataType = 'val2017'
# 保存为新的JSON文件
#output_file = '{}/annotations/instances_{}_with_rbox2.json'.format(dataDir, dataType)

imgDir=r'C:\Users\yangxinyao\Downloads\coco2017\val2017'
output_file=r'C:\Users\yangxinyao\Downloads\coco2017\annotations\instances_val2017.json'
# 可视化部分
# 重新初始化COCO对象，使用新的JSON文件
try:
    # 以utf-8编码格式打开文件
    with open(output_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    coco_new = COCO(output_file)
except FileNotFoundError:
    print(f"错误: 文件 {output_file} 未找到。")
except json.JSONDecodeError:
    print(f"错误: 文件 {output_file} 不是有效的JSON格式。")
except Exception as e:
    print(f"发生未知错误: {e}")

# 获取所有图像的ID
imgIds = coco_new.getImgIds()

# 定义颜色
bbox_color = (255, 0, 0)  # 绿色
rbox_color = (0, 0, 255)  # 蓝色

# 遍历每张图片
for img_id in imgIds:
    img = coco_new.loadImgs(img_id)[0]
    img_path = '{}/{}'.format(imgDir, img['file_name'])
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 获取该图像的注释ID
    annIds = coco_new.getAnnIds(imgIds=img['id'], catIds=coco_new.getCatIds(), iscrowd=None)
    anns = coco_new.loadAnns(annIds)

    for ann in anns:
        # 绘制原bbox
        x, y, w, h = ann['bbox']
        cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), bbox_color, 2)


        """
        # 绘制segmentation
        if ann['iscrowd'] == 0:
            polygons = ann['segmentation']
            for polygon in polygons:
                points = np.array(polygon).reshape(-1, 2).astype(np.int32)
                cv2.polylines(image, [points], True, bbox_color, 2)
        else:
            rle = ann['segmentation']
            # 创建RLE对象
            rle_obj = maskUtils.frPyObjects(rle, rle["size"][0], rle["size"][1])
            # 解码RLE对象为掩码
            mask = maskUtils.decode(rle_obj)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, bbox_color, 2)
            
        # 绘制rbox
                rbox = np.array(ann['rbox']).astype(np.int32)
                cv2.polylines(image, [rbox], True, rbox_color, 2)
        
                # 绘制new_box
                x, y, w, h = ann['new_bbox']
                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), rbox_color, 2)

        """


    # 显示图片
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title(img['file_name'])
    plt.axis('off')
    plt.show()

