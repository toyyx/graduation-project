# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


class Sam(nn.Module):
    mask_threshold: float = 0.0 # 掩码阈值，用于将预测的掩码转换为二进制掩码
    image_format: str = "RGB" # 图像的格式，默认为 RGB

    def __init__(
        self,
        image_encoder: ImageEncoderViT,  # 图像编码器，用于将输入图像编码为图像嵌入，以便高效地进行掩码预测
        prompt_encoder: PromptEncoder,  # 提示编码器，用于对各种类型的输入提示进行编码
        mask_decoder: MaskDecoder,# 掩码解码器，根据图像嵌入和编码后的提示预测掩码
        pixel_mean: List[float] = [123.675, 116.28, 103.53],  # 用于对输入图像像素进行归一化的均值，默认值适用于常见的图像数据集
        pixel_std: List[float] = [58.395, 57.12, 57.375],# 用于对输入图像像素进行归一化的标准差，默认值适用于常见的图像数据集
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.#SAM 模型根据输入图像和提示信息预测物体的掩码。

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.#  用于将图像编码为图像嵌入的骨干网络，便于高效地进行掩码预测
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.# 对各种类型的输入提示进行编码
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings#根据图像嵌入和编码后的提示预测掩码
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.# 用于对输入图像像素进行归一化的均值
          pixel_std (list(float)): Std values for normalizing pixels in the input image.# 用于对输入图像像素进行归一化的标准差
        """
        super().__init__()
        self.image_encoder = image_encoder # 保存图像编码器
        self.prompt_encoder = prompt_encoder# 保存提示编码器
        self.mask_decoder = mask_decoder# 保存掩码解码器
        # 将像素均值、标准差注册为缓冲区，以便在模型保存和加载时一起保存和加载
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)# 并将其形状调整为 (通道数, 1, 1) 以便于广播
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)# 并将其形状调整为 (通道数, 1, 1) 以便于广播

    @property
    def device(self) -> Any:
        return self.pixel_mean.device # 返回模型所在的设备，通过像素均值所在的设备来确定

    @torch.no_grad()
    def forward(
        self,
        batched_input: List[Dict[str, Any]],  # 输入的批量数据，是一个列表，每个元素是一个字典，包含图像和提示信息
        multimask_output: bool,  # 模型是否应该预测多个消歧掩码，还是返回单个掩码
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.
        #从提供的图像和提示信息端到端地预测掩码。
        #如果提示信息事先未知，建议使用 SamPredictor 而不是直接调用该模型。

        Arguments:
          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
            # 输入图像的列表，每个元素是一个字典，包含以下键。如果某个提示键不存在，可以省略。
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
                # 图像，格式为 3xHxW 的 torch 张量，已经过预处理以适合输入到模型中
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
                # 图像在变换之前的原始尺寸，格式为 (高度, 宽度)
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
                # 该图像的批量点提示，形状为 BxNx2，已经转换到模型的输入坐标系中
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
                #点提示的批量标签，形状为 BxN
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
                # 批量的框输入，形状为 Bx4，已经转换到模型的输入坐标系中
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
                #模型的批量掩码输入，格式为 Bx1xHxW
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.
            # 模型是否应该预测多个消歧掩码，还是返回单个掩码

        Returns:
          (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
            # 输入图像的列表，每个元素是一个字典，包含以下键
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input prompts,
                C is determined by multimask_output, and (H, W) is the
                original size of the image.
                #批量的二进制掩码预测结果，形状为 BxCxHxW，其中 B 是输入提示的数量，
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
                #模型对掩码质量的预测，形状为 BxC
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.
                #低分辨率的对数概率，形状为 BxCxHxW，其中 H=W=256。可以作为掩码输入传递给后续的预测迭代
        """
        # 对输入的批量图像进行预处理，并在第 0 维（批量维度）上堆叠
        input_images = torch.stack([self.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = self.image_encoder(input_images)# 将预处理后的图像输入到图像编码器中，得到图像嵌入

        outputs = []# 初始化一个空列表，用于存储每个图像的输出结果
        for image_record, curr_embedding in zip(batched_input, image_embeddings): # 遍历每个输入图像及其对应的图像嵌入
            if "point_coords" in image_record:# 如果输入中包含点提示的坐标
                points = (image_record["point_coords"], image_record["point_labels"])# 提取点提示的坐标和标签
            else:
                points = None# 否则，将点提示设为 None
            sparse_embeddings, dense_embeddings = self.prompt_encoder(  # 使用提示编码器对提示信息进行编码，得到稀疏嵌入和密集嵌入
                points=points,
                boxes=image_record.get("boxes", None),
                masks=image_record.get("mask_inputs", None),
            )
            low_res_masks, iou_predictions = self.mask_decoder( # 使用掩码解码器根据图像嵌入、位置编码、稀疏嵌入和密集嵌入预测低分辨率掩码和掩码质量
                image_embeddings=curr_embedding.unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            masks = self.postprocess_masks( # 对低分辨率掩码进行后处理，将其调整到输入图像的原始尺寸
                low_res_masks,
                input_size=image_record["image"].shape[-2:],
                original_size=image_record["original_size"],
            )
            masks = masks > self.mask_threshold# 根据掩码阈值将掩码转换为二进制掩码
            outputs.append( # 将预测的掩码、掩码质量预测和低分辨率对数概率添加到输出列表中
                {
                    "masks": masks,
                    "iou_predictions": iou_predictions,
                    "low_res_logits": low_res_masks,
                }
            )
        return outputs # 返回所有输入图像的输出结果

    def postprocess_masks(
        self,
        masks: torch.Tensor, # 从掩码解码器输出的批量掩码，形状为 (B, C, H, W)，B 是批量大小，C 是掩码通道数，H 和 W 是掩码的高度和宽度
        input_size: Tuple[int, ...],# 输入到模型的图像尺寸，格式为 (H, W)，用于去除填充
        original_size: Tuple[int, ...], # 在调整大小以输入到模型之前，图像的原始尺寸，格式为 (H, W)
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.
        移除填充并将掩码上采样到原始图像的尺寸。
        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.# 批量掩码，格式为 BxCxHxW，其中 (H, W) 是原始图像的尺寸
        """
        # 将掩码上采样到图像编码器期望的输入尺寸（通常是一个固定的正方形尺寸）
        # 使用双线性插值方法进行上采样，align_corners=False 表示不将角点对齐
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        # 去除之前为了将图像调整为正方形而添加的填充部分
        # 只保留与输入图像实际尺寸对应的部分
        masks = masks[..., : input_size[0], : input_size[1]]
        # 将去除填充后的掩码上采样到原始图像的尺寸
        # 同样使用双线性插值方法，align_corners=False 表示不将角点对齐
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks # 返回处理后的批量掩码

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # 对像素值进行归一化处理，并将图像填充为正方形输入
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std # 对图像的像素值进行归一化，减去像素均值并除以像素标准差

        # Pad
        h, w = x.shape[-2:]# 获取图像的高度和宽度
        padh = self.image_encoder.img_size - h # 计算在高度方向上需要填充的像素数，以达到图像编码器期望的输入尺寸
        padw = self.image_encoder.img_size - w  # 计算在宽度方向上需要填充的像素数，以达到图像编码器期望的输入尺寸
        # 使用 PyTorch 的 F.pad 函数对图像进行填充
        # 填充顺序为 (左, 右, 上, 下)，这里只在右侧和底部进行填充
        x = F.pad(x, (0, padw, 0, padh))
        return x # 返回预处理后的图像
