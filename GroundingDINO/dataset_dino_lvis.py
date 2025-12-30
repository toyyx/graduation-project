import os
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import json

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def to_tensor(ann):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if "bbox" in ann:
        ann["bbox"] = torch.as_tensor(ann["bbox"], dtype=torch.float32).reshape(-1, 4)
    if "new_bbox" in ann:
        ann["new_bbox"] = torch.as_tensor(ann["new_bbox"], dtype=torch.float32).reshape(-1, 4)
    if "rbox" in ann:
        ann["rbox"] = torch.as_tensor(ann["rbox"], dtype=torch.float32).reshape(-1, 4, 2)
    if "fix" in ann:
        ann["fix"] = torch.as_tensor(ann["fix"], dtype=torch.float32).reshape(-1, 4)
    if "ratio" in ann:
        ann["ratio"] = torch.as_tensor(ann["ratio"], dtype=torch.float32).reshape(-1, 1)
    if "caption" in ann:
        ann["caption"] =preprocess_caption(ann["caption"])
    return ann

def coco_url_to_filename(coco_url):
    parts = coco_url.split('/')
    relative_img_path = '/'.join(parts[-2:])
    return relative_img_path

class LVISDataset_dino_lvis(Dataset):
    def __init__(self, ann_file, image_dir, transform=None, max_data_size=None):
        """
        初始化数据集类
        :param ann_file: 新生成的COCO标注文件路径
        :param data_dir: 图像数据所在目录
        :param transform: 图像预处理转换操作
        """

        self.ann_file = ann_file
        self.image_dir = image_dir
        self.transform = transform
        self.max_data_size = max_data_size

        # 加载标注文件
        with open(ann_file, 'r') as f:
            self.lvis_data = json.load(f)

        # 初始化COCO对象
        #coco = COCO(ann_file)

        # 加载所有图片ID
        #img_ids = coco.getImgIds()
        #self.img_ids = self.lvis_api.get_img_ids()

        # 加载类别信息，用于获取类别名称
        #categories = coco.loadCats(coco.getCatIds())
        self.category_id_to_name = {cat['id']: cat['synset'] for cat in  self.lvis_data['categories']}

        # 为了方便根据image_id查找图片信息，将图片信息存储在字典中
        self.image_id_to_name = {img['id']: coco_url_to_filename(img['coco_url']) for img in self.lvis_data['images']}
        self.annotations = self.lvis_data['annotations']

        # 根据 max_data_size 截取数据集
        if max_data_size is not None and len(self.annotations) > max_data_size:
            self.annotations = self.annotations[:max_data_size]

    def __len__(self):
        """
        返回数据集的长度，即标注信息的数量
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        根据索引获取一条标注信息，返回处理后的图像和标注信息字典
        :param idx: 索引
        :return: image（处理后的图像）, target（标注信息字典）
        """
        ann = self.annotations[idx]
        filename = self.image_id_to_name[ann['image_id']]
        img_path = os.path.join(self.image_dir, filename)

        # 打开图像
        image = Image.open(img_path).convert('RGB')
        #print(image.width)
        #print(image.height)
        w, h = image.size
        #print(w)
        #print(h)
        target = to_tensor(ann.copy())
        if self.transform:
            # 对图像进行预处理
            image,target = self.transform(image,target) ####

        return image, w, h, preprocess_caption(self.category_id_to_name[ann['category_id']]), target




