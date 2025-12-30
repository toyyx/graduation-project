import os.path

import cv2
import matplotlib.pyplot as plt
from lvis import LVIS
import numpy as np

# 定义图片根路径，需要根据实际情况修改
image_root_path = 'your_image_root_path'

# 初始化 LVIS API
lvis_api = LVIS(ann_file='path/to/lvis_v1_val.json')

# 获取所有图像的 ID
img_ids = lvis_api.get_img_ids()

# 遍历每张图片
for img_id in img_ids:
    # 获取该图像的信息
    img_info = lvis_api.load_imgs([img_id])[0]

    # 从 coco_url 中提取图片路径
    coco_url = img_info['coco_url']
    parts = coco_url.split('/')
    relative_img_path = '/'.join(parts[-2:])
    full_img_path= os.path.join(image_root_path,relative_img_path)

    # 读取图像
    try:
        image = cv2.imread(full_img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error reading image {full_img_path}: {e}")
        continue

    # 获取该图像的所有标注
    ann_ids = lvis_api.get_ann_ids(img_ids=[img_id])
    anns = lvis_api.load_anns(ann_ids)

    # 可视化标注
    for ann in anns:
        # 绘制边界框
        bbox = ann['bbox']
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 获取类别名称
        category_id = ann['category_id']
        category = lvis_api.load_cats([category_id])[0]['name']

        # 绘制类别名称
        cv2.putText(image, category, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 绘制分割掩码
        mask = lvis_api.ann_to_mask(ann)
        color = np.random.randint(0, 256, 3, dtype=np.uint8)
        mask_image = np.zeros_like(image)
        mask_image[mask == 1] = color
        image = cv2.addWeighted(image, 1, mask_image, 0.5, 0)

    # 显示图像
    plt.imshow(image)
    plt.axis('off')
    plt.show()