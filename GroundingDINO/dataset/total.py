import sys

from pycocotools.coco import COCO

# 数据集路径
dataDir = r'C:\Users\yangxinyao\Downloads\coco2017'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

# 初始化 COCO api
coco = COCO(r'C:\Users\yangxinyao\Downloads\coco2017\annotations\instances_train2017_with_rbox.json')

# 获取所有图像的 ID
imgIds = coco.getImgIds()

# 遍历每张图片
for img_id in imgIds:
    img = coco.loadImgs(img_id)[0]
    # 获取该图像的注释 ID
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=coco.getCatIds(), iscrowd=None)
    anns = coco.loadAnns(annIds)

    # 统计一张图片的标注信息数量
    total_annotations = len(anns)

    # 统计标注类别有几种以及每种几条标注信息
    category_count = {}
    for ann in anns:
        category_id = ann['category_id']
        category_name = coco.loadCats(category_id)[0]['name']
        if category_name not in category_count:
            category_count[category_name] = 1
        else:
            category_count[category_name] += 1

    num_categories = len(category_count)

    print(f"图片 ID: {img_id}")
    print(f"标注信息数量: {total_annotations}")
    print(f"标注类别数量: {num_categories}")
    print("每种标注类别的标注信息数量:")
    for category, count in category_count.items():
        #if count != 1:
            #print(f"wrong！！！！！！！！！！！！")
            #sys.exit()
        print(f"{category}: {count}")
    print()