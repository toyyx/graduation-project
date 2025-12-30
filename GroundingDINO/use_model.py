import cv2
import numpy as np
import torch
import torch.nn as nn
#from mmdet.models.losses.giou_loss import GIoULoss
import torchvision.transforms as T
from PIL import Image
"""
class GlidingVertexLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()  # 或 Focal Loss
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        self.giou_loss = GIoULoss(reduction='none')
        self.alpha = alpha  # 分类损失权重
        self.beta = beta  # 回归损失权重

    def forward(self, pred_boxes, pred_scores, gt_boxes, gt_labels):
       
        pred_boxes: Tensor [B, N, 8] (4 顶点坐标 x1,y1,x2,y2,x3,y3,x4,y4)
        pred_scores: Tensor [B, N, C]
        gt_boxes: Tensor [B, M, 8]
        gt_labels: Tensor [B, M]
      
        # 1. 分类损失
        cls_loss = self.cls_loss(pred_scores, gt_labels)

        # 2. 顶点回归损失
        reg_loss = self.reg_loss(pred_boxes, gt_boxes).mean()

        # 3. GIoU 损失（需将顶点转换为旋转框）
        giou_loss = self.giou_loss(vertices_to_rotated_boxes(pred_boxes),
                                   vertices_to_rotated_boxes(gt_boxes)).mean()

        total_loss = self.alpha * cls_loss + self.beta * (reg_loss + giou_loss)
        return total_loss
"""



def vertices_to_rotated_boxes(vertices):
    # 实现顶点到旋转框的转换（需自定义或调用现有库）
    # 返回格式 [cx, cy, w, h, theta]
    pass

from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
import groundingdino.datasets.transforms as T

from huggingface_hub import hf_hub_download

def init_model():
    model_config_path = "./groundingdino/config/GroundingDINO_rbox_SwinT_OGC.py"
    ckpt_file_path = r"C:\Users\yangxinyao\Downloads\coco2017\save_model\model_epoch_3_best_loss_1.3029.pth"

    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    #args.device = 'cpu'
    checkpoint = torch.load(ckpt_file_path, map_location='cpu')

    log = model.load_state_dict(checkpoint, strict=False)

    print("Model loaded from {} \n => {}".format(ckpt_file_path, log))
    #_ = model.eval()

    model.eval()
    return model

def load_image(image_pil):
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image

def image_transform_grounding_for_vis(init_image):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
    ])
    image, _ = transform(init_image, None) # 3, h, w
    return image

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def test(model,image,caption,box_threshold: float,text_threshold: float,):
    caption1 = preprocess_caption(caption=caption)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption1])
    # 从模型输出中提取预测的类别 logits，并将其移动到 CPU 上，然后应用 sigmoid 函数将其转换为概率值
    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
    prediction_fix = outputs["pred_fix"].cpu()[0]  # prediction_boxes.shape = (nq, 4)
    prediction_ratio = outputs["pred_ratio"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold  # 创建一个掩码，筛选出类别概率最大值大于边界框置信度阈值的预测结果
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
    fixs = prediction_fix[mask]
    ratios = prediction_ratio[mask]

    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)

    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]

    return boxes, fixs,ratios,logits.max(dim=1)[0], phrases  # 返回筛选后的边界框、边界框置信度（类别概率最大值）和匹配的文本短语

def draw_rotated_boxes(image_pil, boxes, fixs, ratios,logits,phrases):
    # 读取图片
    # 将 PIL.Image 转换为 numpy.ndarray
    image = np.array(image_pil)
    height, width, _ = image.shape
    #height, width = image.shape[:2]

    # 定义扩展系数
    expansion_factor = 1.2

    for i in range(len(boxes)):
        # 解析边界框信息
        cx, cy, w, h = boxes[i]
        # 将归一化的坐标转换为像素坐标
        cx = cx * width
        cy = cy * height
        w = w * width * expansion_factor
        h = h * height * expansion_factor


        # 计算四个顶点的初始坐标
        top_left = (max(cx - w * 0.5,0), max(cy - h * 0.5,0))
        top_right = (min(cx + w * 0.5,width-1), max(cy - h * 0.5,0))
        bottom_right = (min(cx + w * 0.5,width-1), min(cy + h * 0.5,height-1))
        bottom_left = (max(cx - w * 0.5,0), min(cy + h * 0.5,height-1))

        # 绘制原本的框
        pts_original = np.array([top_left, top_right, bottom_right, bottom_left],np.int32)
        pts_original = pts_original.reshape((-1, 1, 2))
        cv2.polylines(image, [pts_original], True, (255, 0, 0), 2)

        if ratios[i] <= 0.8:

            # 应用边界框修正
            fix_top_left, fix_top_right, fix_bottom_right, fix_bottom_left = fixs[i]
            top_left = (
                min(top_left[0] + fix_top_left * w,width-1),
                top_left[1]
            )
            top_right = (
                top_right[0],
                min(top_right[1] + fix_top_right * h,height-1)
            )
            bottom_right = (
                max(bottom_right[0] - fix_bottom_right * w,0),
                bottom_right[1]
            )
            bottom_left = (
                bottom_left[0],
                max(bottom_left[1] - fix_bottom_left * h,0)
            )

            # 绘制旋转框
            pts = np.array([top_left, top_right, bottom_right, bottom_left], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 0, 255), 2)

            # 显示匹配的文本短语
            cv2.putText(image, f'{phrases[i]}-{logits[i]}', (int(top_left[0]), int(top_left[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        # 显示图片
    cv2.imshow('Image with Rotated Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



model = init_model()


image_path=r'C:\Users\yangxinyao\Downloads\coco2017\train2017\000000157099.jpg'

init_image = Image.open(image_path).convert("RGB")  # load image
_, image=load_image(init_image)
image_pil=image_transform_grounding_for_vis(init_image)

caption='dog'

boxes, fixs, ratios, logits, phrases=test(model,image,caption,0.25,0.1)


draw_rotated_boxes(image_pil, boxes, fixs,ratios,logits,phrases)



