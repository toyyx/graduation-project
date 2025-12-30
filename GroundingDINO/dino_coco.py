import json
import os
from groundingdino.models import build_model
import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import groundingdino.datasets.transforms as T
from dataset_dino_coco import COCODataset_dino_coco
from groundingdino.util import box_ops
import logging
from groundingdino.util.misc import nested_tensor_from_tensor_list
from scipy.optimize import linear_sum_assignment
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict

def init_logger(filename):
    # 创建第一个日志记录器
    logger1 = logging.getLogger(filename)
    logger1.setLevel(logging.INFO)

    # 创建第一个日志记录器的文件处理器
    file_handler1 = logging.FileHandler(filename)
    file_handler1.setLevel(logging.INFO)

    # 创建第一个日志记录器的格式化器
    formatter1 = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler1.setFormatter(formatter1)

    # 将处理器添加到第一个日志记录器
    logger1.addHandler(file_handler1)

    return logger1

logger1=init_logger('dino_coco.log')

# 计算广义交并比（GIoU），输入格式为 xyxy
def giou_matrix(boxes1, boxes2):
    """
    boxes1: [N, 4] (x1, y1, x2, y2)
    boxes2: [M, 4] (x1, y1, x2, y2)
    """
    x1_1, y1_1, x2_1, y2_1 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x1_2, y1_2, x2_2, y2_2 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # 计算交集的坐标
    inter_x1 = torch.max(x1_1[:, None], x1_2[None, :])
    inter_y1 = torch.max(y1_1[:, None], y1_2[None, :])
    inter_x2 = torch.min(x2_1[:, None], x2_2[None, :])
    inter_y2 = torch.min(y2_1[:, None], y2_2[None, :])

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h

    # 计算并集的面积
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1[:, None] + area2[None, :] - inter_area

    # 计算包含两个框的最小矩形的坐标
    enclosing_x1 = torch.min(x1_1[:, None], x1_2[None, :])
    enclosing_y1 = torch.min(y1_1[:, None], y1_2[None, :])
    enclosing_x2 = torch.max(x2_1[:, None], x2_2[None, :])
    enclosing_y2 = torch.max(y2_1[:, None], y2_2[None, :])

    enclosing_w = enclosing_x2 - enclosing_x1
    enclosing_h = enclosing_y2 - enclosing_y1
    enclosing_area = enclosing_w * enclosing_h

    # 计算GIoU
    giou_value = inter_area / (union_area + 1e-6) - (enclosing_area - union_area) / (enclosing_area + 1e-6)
    return 1 - giou_value


# 定义一个函数来计算成本矩阵
def calculate_cost_matrix(outputs, targets):
    pred_logits = outputs["pred_logits"].cpu()
    pred_boxes = outputs["pred_boxes"].cpu()
    pred_fix = outputs["pred_fix"].cpu()
    #pred_ratio = outputs["pred_ratio"].cpu()

    batch_size = pred_logits.size(0)
    num_queries = pred_logits.size(1)
    num_targets = [len(target["bbox"]) for target in targets]

    cost_matrices = []
    for b in range(batch_size):
        #cost_matrix = torch.zeros((num_queries, num_targets[b]), device=device)

        # 计算框成本
        pred_boxes_b = pred_boxes[b]
        target_boxes = targets[b]["bbox"] #xywh

        # 这里简单使用 L1 距离作为框成本
        box_l1_cost = torch.cdist(pred_boxes_b, target_boxes, p=1) #####

        box_giou=giou_matrix(box_ops.box_cxcywh_to_xyxy(pred_boxes_b),box_ops.box_cxcywh_to_xyxy(target_boxes))

        # 综合成本
        cost_matrix = box_l1_cost + box_giou #nm

        cost_matrices.append(cost_matrix)

    return cost_matrices

# 定义函数来获取每个批次的 row_indices, col_indices 对应关系
def get_matching_indices(cost_matrices):
    batch_size = len(cost_matrices)
    matching_indices_list = []
    for b in range(batch_size):
        cost_matrix = cost_matrices[b].cpu().detach().numpy()
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        matching_indices_list.append((row_indices, col_indices))
    return matching_indices_list



# 定义函数来获取每个批次的 row_indices, col_indices 对应关系

def init_model(checkpoint_path=None):
    model_config_path = "./groundingdino/config/GroundingDINO_rbox_SwinT_OGC.py"
    ckpt_file_path = "./weights/groundingdino_swint_ogc.pth"
    args = SLConfig.fromfile(model_config_path)
    model = build_model(args)
    # args.device = 'cpu'

    if checkpoint_path is None:
        checkpoint = torch.load(ckpt_file_path, map_location='cpu')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(ckpt_file_path, log))
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        log = model.load_state_dict(checkpoint, strict=False)
        print("Model loaded from {} \n => {}".format(checkpoint_path, log))

    return model

def init_dataloder(ann_file,image_dir,batch_size,max_data_size=None):
    def deal_batch_data(batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0])
        return tuple(batch)

    # 定义图像预处理操作
    transform = T.Compose([
            T.RandomResize([512], max_size=1333),#800
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = COCODataset_dino_coco(ann_file, image_dir, transform=transform, max_data_size=max_data_size)
    dataloder= DataLoader(dataset, batch_size=batch_size, collate_fn=deal_batch_data)
    return dataloder


def update_coco_data(coco_data,ann_id,pred_bbox,pred_fix):
    # 更新标注信息
    for ann in coco_data['annotations']:
        if ann['id'] == ann_id:
            assert 'dino_bbox' not in ann
            ann['dino_bbox'] = pred_bbox
            ann['dino_fix'] = pred_fix
            logger1.info(ann)
            break

def save_coco_data(coco_data):
    # 保存新的 COCO 数据集
    new_annotation_path = '/root/autodl-tmp/coco2017/annotations/instances_val2017_dino.json'
    with open(new_annotation_path, 'w') as f:
        json.dump(coco_data, f)

def update_coco_with_grounding_dino(coco_annotation_path, image_dir, checkpoint_path):
    # 加载 COCO 数据集
    with open(coco_annotation_path, 'r') as f:
        coco_data = json.load(f)
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载 Grounding DINO 模型
    model = init_model(checkpoint_path)
    model.to(device)
    dataloder=init_dataloder(coco_annotation_path,image_dir,1,None)
    num_batch=len(dataloder)
    

    model.eval()
    with torch.no_grad():
        for batch_idx, (samples, w, h, captions, targets) in tqdm(enumerate(dataloder), total=num_batch,desc=f'dino_coco ==> '):
            samples = samples.to(device)
            w=w[0]
            h=h[0]
            captions=captions[0]
            #print(captions)
            #print(targets)
            captions = [captions]
            outputs = model(samples, captions=captions)

            # 计算成本矩阵
            cost_matrices = calculate_cost_matrix(outputs, targets)
            # 获取每个批次的 row_indices, col_indices 对应关系
            matching_indices_list = get_matching_indices(cost_matrices)

            row_indices, col_indices = matching_indices_list[0]

            pred_boxes = outputs["pred_boxes"].cpu()
            pred_fix = outputs["pred_fix"].cpu()
            matched_pred_boxes = pred_boxes[0][row_indices][0]#4 cxcywh
            matched_pred_fix = pred_fix[0][row_indices][0]
            pred_bbox=torchvision.ops.box_convert(matched_pred_boxes, 'cxcywh', 'xywh')
            pred_bbox=[pred_bbox[0]*w,pred_bbox[1]*h,pred_bbox[2]*w,pred_bbox[3]*h]
            pred_fix=matched_pred_fix
            update_coco_data(coco_data,targets[0]['id'], [tensor.tolist() for tensor in pred_bbox],pred_fix.tolist() )

        save_coco_data(coco_data)


# 示例调用
coco_annotation_path = '/root/autodl-tmp/coco2017/annotations/instances_val2017.json'
image_dir = '/root/autodl-tmp/coco2017/val2017'
checkpoint_path= r'./checkpoint/model_epoch_19_lr6.pth'

update_coco_with_grounding_dino(coco_annotation_path, image_dir, checkpoint_path)
