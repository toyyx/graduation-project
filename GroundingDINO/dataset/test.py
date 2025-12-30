import os
import cv2
import numpy as np
import matplotlib
from matplotlib.patches import Rectangle

# 修改后端为 Agg，适合在无图形界面的环境中保存图片
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

# 数据集路径
dataDir = r'C:\Users\yangxinyao\Downloads\coco2017'
dataType = 'train2017'
annFile = os.path.join(dataDir, 'annotations', f'stuff_{dataType}.json')

try:
    # 初始化COCO api
    coco = COCO(annFile)
except FileNotFoundError:
    print(f"未找到注释文件，请检查路径 {annFile} 是否正确。")
    exit(1)
except Exception as e:
    print(f"初始化COCO API时出现错误: {e}")
    exit(1)

# 获取所有类别
cats = coco.loadCats(coco.getCatIds())
cat_id_to_name = {cat['id']: cat['name'] for cat in cats}
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

# 获取所有图像的ID
imgIds = coco.getImgIds()

if not imgIds:
    print("未找到图像ID，请检查数据集路径和数据完整性。")
    exit(1)

# 遍历图像
for img_id in imgIds:
    img = coco.loadImgs(img_id)[0]

    # 加载图像
    img_path = '{}/{}/{}'.format(dataDir, dataType, img['file_name'])
    try:
        I = cv2.imread(img_path)
        if I is None:
            raise FileNotFoundError
        I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
    except FileNotFoundError:
        print(f"未找到图像文件，请检查路径 {img_path} 是否正确。")
        continue
    except Exception as e:
        print(f"加载图像时出现错误: {e}")
        continue

    # 获取该图像的注释ID
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=coco.getCatIds(), iscrowd=None)
    anns = coco.loadAnns(annIds)

    if not anns:
        print(f"图像 {img['file_name']} 没有可用的注释。")
    else:
        # 显示图像
        plt.rcParams['figure.figsize'] = (20.0, 20.0)
        plt.imshow(I)

        for ann in anns:
            # 显示边界框
            x, y, w, h = ann['bbox']
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)

            # 显示类别名称
            cat_name = cat_id_to_name[ann['category_id']]
            plt.text(x, y - 10, cat_name, color='r', fontsize=12)

            # 显示掩码，随机生成颜色
            mask = coco.annToMask(ann)
            random_color = np.random.rand(3)
            rgba_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
            rgba_mask[mask > 0] = np.append(random_color, 0.5)
            plt.imshow(rgba_mask)

        # 显示图片名
        plt.title(img['file_name'])
        plt.axis('off')
        plt.show(block=False)
        plt.pause(0.01)
        input("按任意键显示下一张图像...")
        plt.close()

