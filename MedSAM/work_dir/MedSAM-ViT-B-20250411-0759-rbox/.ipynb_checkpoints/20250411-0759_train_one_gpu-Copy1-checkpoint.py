# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
freeze prompt image encoder
"""
import cv2
# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

join = os.path.join
from tqdm import tqdm
from skimage import transform
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import shutil
import glob

def calculate_iou(pred_mask, gt_mask):
    batch_size = pred_mask.size(0)
    batch_ious = []
    for i in range(batch_size):
        pred = pred_mask[i].bool()
        gt = gt_mask[i].bool()
        intersection = (pred & gt).sum().item()
        union = (pred | gt).sum().item()
        if union == 0:
            iou = 0
        else:
            iou = intersection / union
        batch_ious.append(iou)
    average_iou = sum(batch_ious) / len(batch_ious)
    return average_iou

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

    # 计算 rbox 与 bbox 面积的比值
    rbox_area = cv2.contourArea(rbox.astype(np.int32))
    bbox_area = bbox_width * bbox_height
    ratio = rbox_area / bbox_area if bbox_area > 0 else 0

    return new_bbox_xyxy, fix, ratio

def get_rbox_from_xyxybbox_with_fix(bbox,fix):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    widths = x2-x1
    heights = y2-y1

    top_point=[x1 + widths * fix[0], y1]
    right_point=[x2 , y1 + heights * fix[1]]
    bottom_point=[x2 - widths * fix[2], y2]
    left_point=[x1 , y2 - heights * fix[3]]

    return [top_point,right_point,bottom_point,left_point]

# set seeds
torch.manual_seed(2023)  # 设置随机种子，保证结果可复现
torch.cuda.empty_cache()  # 清空CUDA缓存

# torch.distributed.init_process_group(backend="gloo")
# 设置环境变量，控制多线程计算的线程数
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


# 定义函数，用于在图像上显示掩码
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)  # 随机生成颜色
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])  # 使用默认颜色
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # 将掩码与颜色相乘，得到掩码图像
    ax.imshow(mask_image)  # 在指定的坐标轴上显示掩码图像


# 定义函数，用于在图像上显示边界框
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(  # 在指定的坐标轴上添加矩形框
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )
from matplotlib.patches import Polygon
def show_rbox(rbox, ax):
    """
    该函数用于在指定的坐标轴上绘制由四点坐标表示的矩形框
    :param rbox: 四点坐标，格式为[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    :param ax: matplotlib的坐标轴对象
    """
    # 提取四点坐标中的x坐标和y坐标
    x_coords = [point[0] for point in rbox]
    y_coords = [point[1] for point in rbox]
    # 创建多边形对象，设置边框颜色、填充颜色和线宽
    poly = Polygon(list(zip(x_coords, y_coords)), edgecolor="red", facecolor=(0, 0, 0, 0), lw=2)
    # 将多边形添加到坐标轴上
    ax.add_patch(poly)

# 自定义数据集类，用于加载npy格式的图像和掩码数据
class NpyDataset(Dataset):
    def __init__(self, data_root, bbox_shift=20):
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.gt_path_files = sorted(  # 获取所有掩码文件的路径
            glob.glob(join(self.gt_path, "**/*.npy"), recursive=True)
        )
        self.gt_path_files = [  # 过滤掉那些在图像路径下找不到对应图像的掩码文件
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])  # 加载npy格式的图像，形状为(1024, 1024, 3)，像素值范围为[0, 1]
        img_1024 = np.load(
            join(self.img_path, img_name), "r", allow_pickle=True
        )  # (1024, 1024, 3)
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))  # 将图像的形状转换为(3, H, W)
        assert (  # 确保图像的像素值在[0, 1]范围内
                np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        ), "image should be normalized to [0, 1]"
        gt = np.load(  # 加载对应的掩码，可能包含多个标签
            self.gt_path_files[index], "r", allow_pickle=True
        )  # multiple labels [0, 1,4,5...], (256,256)
        assert img_name == os.path.basename(self.gt_path_files[index]), (  # 确保图像和掩码的文件名一致
                "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        )
        label_ids = np.unique(gt)[1:]  # 获取掩码中所有非零标签的ID
        gt2D = np.uint8(  # 随机选择一个标签，将掩码转换为只包含该标签的二值掩码
            gt == random.choice(label_ids.tolist())
        )  # only one label, (256, 256)
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"  # 确保二值掩码的像素值为0或1
        y_indices, x_indices = np.where(gt2D > 0)  # 获取掩码中所有非零像素的坐标
        # 计算边界框的坐标
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        # add perturbation to bounding box coordinates
        H, W = gt2D.shape  # 对边界框的坐标添加随机扰动
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])

        rbox = get_rbox_from_binary_mask(gt2D, W, H)
        _, fix, _ = calculate_bbox_and_fix(rbox)
        rbox = get_rbox_from_xyxybbox_with_fix(bboxes, fix)
        return (  # 返回图像、掩码、边界框和文件名
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            torch.tensor(bboxes).float(),
            torch.tensor(rbox).float(),
            img_name,
        )


# %% sanity test of dataset class
tr_dataset = NpyDataset("/root/autodl-tmp/MedSAM_data/npy_train/CT_Abd")  # 创建训练数据集实例
tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)  # 创建数据加载器，设置批量大小为8，打乱数据顺序
for step, (image, gt, bboxes, rboxes, names_temp) in enumerate(tr_dataloader):  # 遍历数据加载器，获取一个批次的数据
    print(image.shape, gt.shape, bboxes.shape, rboxes.shape)  # 打印图像、掩码和边界框的形状
    # show the example
    _, axs = plt.subplots(1, 2, figsize=(25, 25))  # 创建一个包含两个子图的画布
    idx = random.randint(0, 7)  # 随机选择一个样本进行可视化
    axs[0].imshow(image[idx].cpu().permute(1, 2, 0).numpy())  # 在第一个子图中显示图像
    show_mask(gt[idx].cpu().numpy(), axs[0])  # 在第一个子图中显示掩码
    show_box(bboxes[idx].numpy(), axs[0])  # 在第一个子图中显示边界框
    show_rbox(rboxes[idx].numpy(), axs[0])
    axs[0].axis("off")  # 关闭第一个子图的坐标轴
    # set title
    axs[0].set_title(names_temp[idx])  # 设置第一个子图的标题
    idx = random.randint(0, 7)  # 随机选择另一个样本进行可视化
    axs[1].imshow(image[idx].cpu().permute(1, 2, 0).numpy())  # 在第二个子图中显示图像
    show_mask(gt[idx].cpu().numpy(), axs[1])  # 在第二个子图中显示掩码
    show_box(bboxes[idx].numpy(), axs[1])  # 在第二个子图中显示边界框
    show_rbox(rboxes[idx].numpy(), axs[1])
    axs[1].axis("off")  # 关闭第二个子图的坐标轴
    # set title
    axs[1].set_title(names_temp[idx])  # 设置第二个子图的标题
    # plt.show()
    plt.subplots_adjust(wspace=0.01, hspace=0)  # 调整子图之间的间距
    plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)  # 保存可视化结果为图片
    plt.close()  # 关闭画布
    break  # 只处理一个批次的数据，然后跳出循环

# 直接指定命令行参数
tr_npy_path = "/root/autodl-tmp/MedSAM_data/npy_train/CT_Abd"
task_name = "MedSAM-ViT-B"
model_type = "vit_b"

load_pretrain = True
pretrain_model_path = ""  ##no use
work_dir = "./work_dir"  ## save model dir
num_epochs = 100
batch_size = 2
num_workers = 0
weight_decay = 0.01
lr = 0.0001
use_wandb = False
use_amp = False
resume = ""  ##checkpoint path
device = "cuda:0"

if use_wandb:  # 如果使用wandb监控训练过程
    import wandb

    wandb.login()  # 登录wandb
    wandb.init(  # 初始化wandb项目
        project=task_name,
        config={
            "lr": lr,
            "batch_size": batch_size,
            "data_path": tr_npy_path,
            "model_type": model_type,
        },
    )

# %% set up model for training
# device = device
run_id = datetime.now().strftime("%Y%m%d-%H%M")  # 获取当前时间，用于生成唯一的运行ID
model_save_path = join(work_dir, task_name + "-" + run_id)  # 构建模型保存路径
device = torch.device(device)


# %% set up model


class MedSAM(nn.Module):  # 自定义MedSAM模型类
    def __init__(
            self,
            image_encoder,
            mask_decoder,
            prompt_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # freeze prompt encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = True
        for param in self.mask_decoder.parameters():
            param.requires_grad = True
        #for name, param in self.prompt_encoder.named_parameters():
            #if 'rbox_point_embeddings' in name:
                #param.requires_grad = True
                #print(f"可训练参数名: {name}")
            #else:
                #param.requires_grad = False

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, box, rbox):
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # 切断 image_embedding 的梯度传播
        #image_embedding = image_embedding.detach()
        
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            if len(box_torch.shape) == 2:
                box_torch = box_torch[:, None, :]  # (B, 1, 4)
            rbox_torch = torch.as_tensor(rbox, dtype=torch.float32, device=image.device)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None,
            boxes=box_torch,
            rboxes=rbox_torch,#rbox_torch
            masks=None,
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        return ori_res_masks


def main():
    os.makedirs(model_save_path, exist_ok=True)  # 创建模型保存目录，如果目录已存在则不会报错
    shutil.copyfile(  # 复制当前脚本到模型保存目录
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )

    sam_model = sam_model_registry[model_type]()

    checkpoint_path = "./medsam_vit_b.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    log = sam_model.load_state_dict(checkpoint, strict=False)
    print("Model loaded from {} \n => {}".format(checkpoint_path, log))
    
    
    # 检查是否存在 missing_keys
    if log.missing_keys:
        print("存在缺失的键，进行额外处理:")
        rbox_path = 'sam_conbine_epoch_78_5lr5_rbox_best.pth'
        checkpoint2 = torch.load(rbox_path, map_location=device)
        sam_model.prompt_encoder.rbox_point_embeddings[0].weight.data = checkpoint2['prompt_encoder.rbox_point_embeddings.0.weight']
        sam_model.prompt_encoder.rbox_point_embeddings[1].weight.data = checkpoint2['prompt_encoder.rbox_point_embeddings.1.weight']
        sam_model.prompt_encoder.rbox_point_embeddings[2].weight.data = checkpoint2['prompt_encoder.rbox_point_embeddings.2.weight']
        sam_model.prompt_encoder.rbox_point_embeddings[3].weight.data = checkpoint2['prompt_encoder.rbox_point_embeddings.3.weight']
    else:
        print("没有缺失的键，模型加载完整。")

    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    medsam_model.train()

    print(  # 打印模型的总参数数量
        "Number of total parameters: ",
        sum(p.numel() for p in medsam_model.parameters()),
    )  # 93735472
    print(  # 打印模型的可训练参数数量
        "Number of trainable parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252

    #img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + list(
        #medsam_model.mask_decoder.parameters()
    #)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, medsam_model.parameters()), lr=lr, weight_decay=weight_decay
    )
    print(  # 获取图像编码器和掩码解码器的参数
        "Number of image encoder and mask decoder parameters: ",
        sum(p.numel() for p in medsam_model.parameters() if p.requires_grad),
    )  # 93729252
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")  # 定义Dice损失函数
    # cross entropy loss
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")  # 定义二元交叉熵损失函数
    # %% train
    #num_epochs = num_epochs
    iter_num = 0
    losses = []
    ious=[]
    best_loss = 1e10
    train_dataset = NpyDataset(tr_npy_path)

    print("Number of training samples: ", len(train_dataset))  # 定义二元交叉熵损失函数
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    start_epoch = 0
    if resume is not None:  # 如果指定了从检查点恢复训练
        if os.path.isfile(resume):
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            medsam_model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if use_amp:  # 如果使用混合精度训练
        scaler = torch.cuda.amp.GradScaler()  # 开始训练循环

    # 开始训练循环
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        total_iou=0
        for step, (image, gt2D, boxes, rboxes, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            boxes_np = boxes.detach().cpu().numpy()
            rboxes_np = rboxes.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)
            if use_amp:  # 如果使用混合精度训练
                ## AMP
                with torch.autocast(device_type="cuda", dtype=torch.float16):  # 如果使用混合精度训练
                    medsam_pred = medsam_model(image, boxes_np,rboxes_np)
                    loss = seg_loss(medsam_pred, gt2D) + ce_loss(
                        medsam_pred, gt2D.float()
                    )
                    iou = calculate_iou(medsam_pred.squeeze(1) > 0.0, gt2D)
                    total_iou += iou
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                medsam_pred = medsam_model(image, boxes_np,rboxes_np)
                iou = calculate_iou(medsam_pred.squeeze(1) > 0.0, gt2D)
                total_iou += iou
                
                loss = seg_loss(medsam_pred, gt2D) + ce_loss(medsam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iter_num += 1

        epoch_loss /= step  # 计算当前轮的平均损失
        total_iou /= step
        losses.append(epoch_loss)  # 将当前轮的平均损失添加到损失列表中
        ious.append(total_iou)
        if use_wandb:
            wandb.log({"epoch_loss": epoch_loss})
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss},iou:{total_iou}'
        )
        ## save the latest model
        checkpoint = {  # 保存最新的模型检查点
            "model": medsam_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "medsam_model_latest.pth"))
        ## save the best model
        if epoch_loss < best_loss:  # 如果当前轮的平均损失小于最佳损失
            best_loss = epoch_loss
            checkpoint = {
                "model": medsam_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "medsam_model_best.pth"))

        # %% plot loss
        plt.plot(losses)  # 绘制损失曲线
        plt.title("Dice + Cross Entropy Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(model_save_path, task_name + "train_loss.png"))
        plt.close()
        
        plt.plot(ious)  # 绘制损失曲线
        plt.title("mIOU")
        plt.xlabel("Epoch")
        plt.ylabel("mIOU")
        plt.savefig(join(model_save_path, task_name + "train_miou.png"))
        plt.close()


if __name__ == "__main__":
    main()
