import json
from pycocotools.coco import COCO
from tqdm import tqdm

# 设定新生成的COCO数据集JSON文件路径
new_ann_file = r'/root/autodl-tmp/coco2017/annotations/instances_val2017_with_rbox.json'

# 初始化COCO对象
coco = COCO(new_ann_file)

# 加载所有图片ID
img_ids = coco.getImgIds()

# 加载类别信息，用于获取类别名称
categories = coco.loadCats(coco.getCatIds())
category_id_to_name = {cat['id']: cat['name'] for cat in categories}

# 加载原始的COCO数据
with open(new_ann_file, 'r') as f:
    coco_data = json.load(f)

# 为了方便根据image_id查找图片信息，将图片信息存储在字典中
image_id_to_name = {img['id']: img['file_name'] for img in coco_data['images']}

# 存储补充信息后的标注
new_annotations = []

# 遍历每张图片
for img_id in tqdm(img_ids, desc="Processing images"):
    # 获取当前图片的所有标注ID
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # 按类别分组标注
    category_anns = {}
    for ann in anns:
        category_id = ann['category_id']
        if category_id not in category_anns:
            category_anns[category_id] = []
        category_anns[category_id].append(ann)

    # 合并同一类别下的标注
    for category_id, ann_list in category_anns.items():
        # 初始化合并后的标注信息
        merged_ann = {
            'id':f'{img_id}-{category_id}',
            'image_id': img_id,
            'filename': image_id_to_name[img_id],
            'category_id': category_id,
            'caption': category_id_to_name[category_id],
            'bbox': [],
            'rbox': [],
            'new_bbox': [],
            'fix': [],
            'ratio': []
        }

        # 合并bbox、rbox、fix、ratio
        for ann in ann_list:
            merged_ann['bbox'].append(ann['bbox'])
            merged_ann['rbox'].append(ann['rbox'])
            merged_ann['new_bbox'].append(ann['new_bbox'])
            merged_ann['fix'].append(ann['fix'])
            merged_ann['ratio'].append(ann['ratio'])

        # 添加合并后的标注到结果中
        new_annotations.append(merged_ann)

# 更新COCO数据中的标注信息
coco_data['annotations'] = new_annotations

# 保存合并后的数据到新的JSON文件
output_file = r'/root/autodl-tmp/coco2017/annotations/instances_val2017_with_rbox_merge.json'
with open(output_file, 'w') as f:
    json.dump(coco_data, f)

print(f"合并后的标注已保存到 {output_file}")
