# -*- coding: utf-8 -*-
# %% import packages
# pip install connected-components-3d
import numpy as np

# import nibabel as nib
import SimpleITK as sitk
import os

join = os.path.join
from skimage import transform
from tqdm import tqdm
import cc3d

# 功能：将nii格式的医学图像及其对应的掩码转换为npz文件
# 医学图像模态，这里指定为CT
# convert nii image to npz files, including original image and corresponding masks
modality = "CT"
anatomy = "Abd"  # anantomy + dataset name  # 解剖部位和数据集名称，这里是腹部相关的数据集
img_name_suffix = "_0000.nii.gz"    # 原始图像文件名后缀
gt_name_suffix = ".nii.gz"# 掩码文件名后缀
prefix = modality + "_" + anatomy + "_"# 生成文件的前缀，结合模态和解剖部位信息

nii_path = "/root/autodl-tmp/MedSAM_data/FLARE22Train/images"  # path to the nii images # 原始nii图像文件所在路径
gt_path = "/root/autodl-tmp/MedSAM_data/FLARE22Train/labels"  # path to the ground truth # 掩码文件所在路径
npy_path = "/root/autodl-tmp/MedSAM_data/npy_train/" + prefix[:-1] # 保存处理后npz文件的路径
os.makedirs(join(npy_path, "gts"), exist_ok=True)# 创建保存掩码npz文件的目录，如果目录已存在则不会报错
os.makedirs(join(npy_path, "imgs"), exist_ok=True)# 创建保存图像npz文件的目录，如果目录已存在则不会报错

image_size = 1024# 图像的目标尺寸
voxel_num_thre2d = 100# 二维切片中，去除小于该像素数量的连通区域的阈值
voxel_num_thre3d = 1000# 三维空间中，去除小于该像素数量的连通区域的阈值

names = sorted(os.listdir(gt_path))# 获取掩码文件路径下的所有文件名，并按字母顺序排序
print(f"ori \# files {len(names)=}")# 打印原始文件数量
names = [# 过滤掉那些在原始图像路径下找不到对应图像的掩码文件
    name
    for name in names
    if os.path.exists(join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"after sanity check \# files {len(names)=}")# 打印经过检查后剩余的文件数量

# set label ids that are excluded   # 要排除的标签ID列表，这里将十二指肠标签排除，因为它在图像中分散，难以用边界框指定
remove_label_ids = [
    12
]  # remove deodenum since it is scattered in the image, which is hard to specify with the bounding box
# 肿瘤标签ID，仅在有多个肿瘤时设置，用于将语义掩码转换为实例掩码
tumor_id = None  # only set this when there are multiple tumors; convert semantic masks to instance masks
# set window level and width
# https://radiopaedia.org/articles/windowing-ct
# 设定CT图像的窗位和窗宽
WINDOW_LEVEL = 40  # only for CT images
WINDOW_WIDTH = 400  # only for CT images

# %% save preprocessed images and masks as npz files
# 遍历前40个文件，剩余的文件可用于验证
for name in tqdm(names[:40]):  # use the remaining 10 cases for validation
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix    # 构建原始图像文件名
    gt_name = name # 掩码文件名
    gt_sitk = sitk.ReadImage(join(gt_path, gt_name))# 使用SimpleITK读取掩码文件
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))# 将SimpleITK图像对象转换为NumPy数组
    # remove label ids
    for remove_label_id in remove_label_ids:# 移除指定标签ID的区域
        gt_data_ori[gt_data_ori == remove_label_id] = 0
    # label tumor masks as instances and remove from gt_data_ori
    if tumor_id is not None: # 如果设置了肿瘤标签ID
        tumor_bw = np.uint8(gt_data_ori == tumor_id)
        gt_data_ori[tumor_bw > 0] = 0 # 提取肿瘤掩码
        # label tumor masks as instances
        tumor_inst, tumor_n = cc3d.connected_components( # 对肿瘤掩码进行三维连通组件分析，标记不同的肿瘤实例
            tumor_bw, connectivity=26, return_N=True
        )
        # put the tumor instances back to gt_data_ori
        gt_data_ori[tumor_inst > 0] = (# 将标记好的肿瘤实例重新添加到原始掩码中
            tumor_inst[tumor_inst > 0] + np.max(gt_data_ori) + 1
        )

    # exclude the objects with less than 1000 pixels in 3D
    gt_data_ori = cc3d.dust( # 排除三维空间中像素数量少于阈值的连通区域
        gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    )
    # remove small objects with less than 100 pixels in 2D slices
    # 遍历每个二维切片
    for slice_i in range(gt_data_ori.shape[0]):
        gt_i = gt_data_ori[slice_i, :, :] # 获取当前切片的掩码
        # remove small objects with less than 100 pixels
        # reason: fro such small objects, the main challenge is detection rather than segmentation
        gt_data_ori[slice_i, :, :] = cc3d.dust( # 排除当前切片中像素数量少于阈值的连通区域
            gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
        )
    # find non-zero slices
    z_index, _, _ = np.where(gt_data_ori > 0) # 找到掩码中不为零的切片索引
    z_index = np.unique(z_index)# 去除重复的索引

    if len(z_index) > 0: # 如果存在不为零的切片
        # crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :] # 裁剪出包含非零掩码的切片
        # load image and preprocess
        img_sitk = sitk.ReadImage(join(nii_path, image_name))# 读取原始图像文件
        image_data = sitk.GetArrayFromImage(img_sitk) # 将SimpleITK图像对象转换为NumPy数组
        # nii preprocess start
        if modality == "CT": # 开始对原始图像进行预处理
            # 计算窗宽窗位对应的上下界
            lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
            upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
            image_data_pre = np.clip(image_data, lower_bound, upper_bound) # 对图像进行裁剪，将像素值限制在上下界之间
            image_data_pre = ( # 对裁剪后的图像进行归一化处理，将像素值缩放到0-255之间
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
        else:
            # 对于非CT图像，计算像素值的0.5%和99.5%分位数作为上下界
            lower_bound, upper_bound = np.percentile(
                image_data[image_data > 0], 0.5
            ), np.percentile(image_data[image_data > 0], 99.5)
            image_data_pre = np.clip(image_data, lower_bound, upper_bound)# 对图像进行裁剪，将像素值限制在上下界之间
            image_data_pre = (# 对裁剪后的图像进行归一化处理，将像素值缩放到0-255之间
                (image_data_pre - np.min(image_data_pre))
                / (np.max(image_data_pre) - np.min(image_data_pre))
                * 255.0
            )
            image_data_pre[image_data == 0] = 0 # 将原始图像中像素值为0的部分在预处理后仍设为0

        image_data_pre = np.uint8(image_data_pre) # 将预处理后的图像数据转换为uint8类型
        img_roi = image_data_pre[z_index, :, :] # 裁剪出包含非零掩码的切片对应的图像
        # 将处理后的图像、掩码和图像间距信息保存为压缩的npz文件
        np.savez_compressed(join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
        # save the image and ground truth as nii files for sanity check;
        # they can be removed
        # 保存处理后的图像和掩码为nii文件，用于检查处理结果，后续可删除
        img_roi_sitk = sitk.GetImageFromArray(img_roi)
        gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
        sitk.WriteImage(
            img_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
        )
        sitk.WriteImage(
            gt_roi_sitk,
            join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
        )
        # 遍历裁剪后的图像的每个切片
        # save the each CT image as npy file
        for i in range(img_roi.shape[0]):
            img_i = img_roi[i, :, :]# 获取当前切片的图像
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)# 将单通道图像转换为三通道图像
            resize_img_skimg = transform.resize(# 使用skimage库对图像进行缩放，调整到目标尺寸
                img_3c,
                (image_size, image_size),
                order=3,
                preserve_range=True,
                mode="constant",
                anti_aliasing=True,
            )
            resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)# 对缩放后的图像进行归一化处理，将像素值缩放到0-1之间
            gt_i = gt_roi[i, :, :]# 获取当前切片的掩码
            resize_gt_skimg = transform.resize( # 使用skimage库对掩码进行缩放，调整到目标尺寸
                gt_i,
                (image_size, image_size),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
            resize_gt_skimg = np.uint8(resize_gt_skimg) # 将缩放后的掩码转换为uint8类型
            assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape # 确保图像和掩码的尺寸一致
            np.save(# 保存处理后的图像切片为npy文件
                join(
                    npy_path,
                    "imgs",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_img_skimg_01,
            )
            np.save(# 保存处理后的掩码切片为npy文件
                join(
                    npy_path,
                    "gts",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_gt_skimg,
            )
