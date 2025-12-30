# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    #将图像的最长边调整为 'target_length'，同时提供调整坐标和框的方法。
    #支持对 NumPy 数组和批量的 PyTorch 张量进行转换。
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length  # 初始化类实例，接收目标最长边长度作为参数

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        #期望输入一个形状为 HxWxC 的 uint8 格式的 NumPy 数组表示的图像。
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)  # 计算调整后的目标尺寸
        return np.array(resize(to_pil_image(image), target_size))  # 将 NumPy 数组转换为 PIL 图像，调整大小，再转换回 NumPy 数组

    def apply_image_return_target_size(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        #期望输入一个形状为 HxWxC 的 uint8 格式的 NumPy 数组表示的图像。
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)  # 计算调整后的目标尺寸
        return np.array(
            resize(to_pil_image(image), target_size)), target_size  # 将 NumPy 数组转换为 PIL 图像，调整大小，再转换回 NumPy 数组

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        #期望输入一个最后一维长度为 2 的 NumPy 数组表示的坐标，需要原始图像的尺寸，格式为 (H, W)。
        """
        old_h, old_w = original_size  # 获取原始图像的高度和宽度
        new_h, new_w = self.get_preprocess_shape(  # 计算调整后的新高度和新宽度
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)  # 深拷贝坐标数组，避免修改原始数据，并转换为浮点数类型
        coords[..., 0] = coords[..., 0] * (new_w / old_w)  # 根据宽度的缩放比例调整坐标的 x 分量
        coords[..., 1] = coords[..., 1] * (new_h / old_h)  # 根据高度的缩放比例调整坐标的 y 分量
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        #期望输入一个形状为 Bx4 的 NumPy 数组表示的框，需要原始图像的尺寸，格式为 (H, W)。
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)  # 将框的数组重塑为 Bx2x2 的形状，调用 apply_coords 方法调整坐标
        return boxes.reshape(-1, 4)  # 将调整后的坐标数组重塑回 Bx4 的形状

    def apply_rboxes(self, rboxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        #期望输入一个形状为 Bx4*2 的 NumPy 数组表示的框，需要原始图像的尺寸，格式为 (H, W)。
        """
        rboxes = self.apply_coords(rboxes.reshape(-1, 4, 2), original_size)  # 将框的数组重塑为 Bx2x2 的形状，调用 apply_coords 方法调整坐标
        return rboxes.reshape(-1, 4, 2)  # 将调整后的坐标数组重塑回 Bx4 的形状

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        #期望输入批量的图像，形状为 BxCxHxW，且为浮点型格式。
        #此转换可能与 apply_image 不完全一致，apply_image 是模型期望的转换方式。
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)  # 计算调整后的目标尺寸
        return F.interpolate(  # 使用双线性插值对图像进行缩放
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
            self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
         期望输入一个最后一维长度为 2 的 PyTorch 张量表示的坐标，需要原始图像的尺寸，格式为 (H, W)。
        """
        old_h, old_w = original_size  # 获取原始图像的高度和宽度
        new_h, new_w = self.get_preprocess_shape(  # 计算调整后的新高度和新宽度
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)  # 深拷贝坐标张量，避免修改原始数据，并转换为浮点型
        coords[..., 0] = coords[..., 0] * (new_w / old_w)  # 根据宽度的缩放比例调整坐标的 x 分量
        coords[..., 1] = coords[..., 1] * (new_h / old_h)  # 根据高度的缩放比例调整坐标的 y 分量
        return coords

    def apply_boxes_torch(
            self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        #期望输入一个形状为 Bx4 的 PyTorch 张量表示的框，需要原始图像的尺寸，格式为 (H, W)。
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2),
                                        original_size)  # 将框的张量重塑为 Bx2x2 的形状，调用 apply_coords_torch 方法调整坐标
        return boxes.reshape(-1, 4)  # 将调整后的坐标张量重塑回 Bx4 的形状

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        #根据输入尺寸和目标最长边长度计算输出尺寸。
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)  # 计算缩放比例
        newh, neww = oldh * scale, oldw * scale  # 计算调整后的高度和宽度
        neww = int(neww + 0.5)  # 对调整后的宽度进行四舍五入取整
        newh = int(newh + 0.5)  # 对调整后的高度进行四舍五入取整
        return (newh, neww)
