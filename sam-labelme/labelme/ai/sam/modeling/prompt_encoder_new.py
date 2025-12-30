# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
import cv2
from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d
from ..utils.transforms import ResizeLongestSide


class PromptEncoder(nn.Module):  # 定义提示编码器类，用于对输入到SAM的掩码解码器的提示进行编码
    def __init__(
            self,
            embed_dim: int,  # 提示的嵌入维度
            image_embedding_size: Tuple[int, int],  # 图像嵌入的空间大小，格式为 (高度, 宽度)
            input_image_size: Tuple[int, int],  # 输入到图像编码器的填充后图像的大小，格式为 (高度, 宽度)
            mask_in_chans: int,  # 用于编码输入掩码的隐藏通道数
            activation: Type[nn.Module] = nn.GELU,  # 编码输入掩码时使用的激活函数
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim  # 保存提示的嵌入维度
        self.input_image_size = input_image_size  # 保存输入图像的大小
        self.image_embedding_size = image_embedding_size  # 保存图像嵌入的空间大小
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)  # 初始化位置编码层，使用随机位置编码，嵌入维度为embed_dim的一半

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners # 定义点嵌入的数量，包括正/负点和2个框的角点
        point_embeddings = [nn.Embedding(1, embed_dim) for i in
                            range(self.num_point_embeddings)]  # 创建点嵌入层列表，每个点嵌入层是一个Embedding层，嵌入维度为embed_dim
        self.point_embeddings = nn.ModuleList(point_embeddings)  # 将点嵌入层列表转换为nn.ModuleList，方便管理

        self.num_rbox_point_embeddings: int = 4  # 4 box corners # 定义点嵌入的数量，包括框的4个角点
        rbox_point_embeddings = [nn.Embedding(1, embed_dim) for i in
                                 range(self.num_rbox_point_embeddings)]  # 创建点嵌入层列表，每个点嵌入层是一个Embedding层，嵌入维度为embed_dim
        self.rbox_point_embeddings = nn.ModuleList(rbox_point_embeddings)  # 将点嵌入层列表转换为nn.ModuleList，方便管理

        self.num_bbox_point_embeddings_origin: int = 2  # 4 box corners # 定义点嵌入的数量，包括框的4个角点
        bbox_point_embeddings_origin = [nn.Embedding(1, embed_dim) for i in range(
            self.num_bbox_point_embeddings_origin)]  # 创建点嵌入层列表，每个点嵌入层是一个Embedding层，嵌入维度为embed_dim
        self.bbox_point_embeddings_origin = nn.ModuleList(bbox_point_embeddings_origin)  # 将点嵌入层列表转换为nn.ModuleList，方便管理

        self.num_rbox_point_embeddings_origin: int = 4  # 4 box corners # 定义点嵌入的数量，包括框的4个角点
        rbox_point_embeddings_origin = [nn.Embedding(1, embed_dim) for i in range(
            self.num_rbox_point_embeddings_origin)]  # 创建点嵌入层列表，每个点嵌入层是一个Embedding层，嵌入维度为embed_dim
        self.rbox_point_embeddings_origin = nn.ModuleList(rbox_point_embeddings_origin)  # 将点嵌入层列表转换为nn.ModuleList，方便管理

        self.not_a_point_embed = nn.Embedding(1, embed_dim)  # 定义一个特殊的嵌入层，表示不是点的情况

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])  # 计算掩码输入的大小，是图像嵌入大小的4倍
        self.mask_downscaling = nn.Sequential(  # 定义掩码下采样模块，将输入掩码进行下采样并转换为嵌入维度
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            # 第一个卷积层，输入通道为1，输出通道为mask_in_chans的四分之一，卷积核大小为2，步长为2
            LayerNorm2d(mask_in_chans // 4),  # 二维层归一化层，对卷积层的输出进行归一化
            activation(),  # 激活函数层
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            # 第二个卷积层，输入通道为mask_in_chans的四分之一，输出通道为mask_in_chans，卷积核大小为2，步长为2
            LayerNorm2d(mask_in_chans),  # 二维层归一化层
            activation(),  # 激活函数层
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),  # 第三个卷积层，输入通道为mask_in_chans，输出通道为embed_dim，卷积核大小为1
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)  # 定义一个特殊的嵌入层，表示没有掩码的情况

        self.rbox_mask_embed = nn.Embedding(1, embed_dim)  # 定义一个特殊的嵌入层，表示没有掩码的情况

    def get_dense_pe(self) -> torch.Tensor:
        # 返回用于对点提示进行编码的位置编码，
        # 应用于对一组密集点的形状进行图像编码。
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        # 调用位置编码层，得到图像嵌入大小的位置编码
        # 并在第0维添加一个维度，将其形状变为 1x(embed_dim)x(embedding_h)x(embedding_w)
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def compute_pe_for_sampled_points(self, coords):
        """
        为给定的采样点坐标计算位置编码
        :param coords: 采样点坐标，形状为 (b, n, 2)，格式为 (h, w)
        :param h: 图像的高度
        :param w: 图像的宽度
        :return: 位置编码结果，形状为 (b, n, 2 * num_pos_feats)
        """
        with torch.no_grad():
            if isinstance(coords, torch.Tensor):
                # 转换坐标顺序 (h, w) 到 (w, h)
                normalized_coords = torch.stack([coords[..., 1], coords[..., 0]], dim=-1)  # bn2

                return self.pe_layer._pe_encoding(normalized_coords)  # bnd
            elif isinstance(coords, list):
                result = []
                for coord in coords:
                    normalized_coord = torch.stack([coord[..., 1], coord[..., 0]], dim=-1)  # n2
                    result.append(self.pe_layer._pe_encoding(normalized_coord))  # nd
                return result

    def _embed_points(
            self,
            points: torch.Tensor,  # 输入的点坐标张量，形状通常为 (batch_size, num_points, 2)，其中 2 表示点的 (x, y) 坐标
            labels: torch.Tensor,  # 每个点对应的标签张量，形状为 (batch_size, num_points)，标签值可以为 -1、0、1 等，分别表示不同的点类型
            pad: bool,  # 是否对输入的点进行填充
    ) -> torch.Tensor:  # 嵌入后的点提示张量，形状与输入的 points 张量相关，最后一维为嵌入维度
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel # 将点坐标偏移 0.5，使其移动到像素的中心位置
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2),
                                        device=points.device)  # 创建一个全零的填充点张量，形状为 (batch_size, 1, 2)，表示在每个样本中添加一个填充点
            padding_label = -torch.ones((labels.shape[0], 1),
                                        device=labels.device)  # 创建一个全 -1 的填充标签张量，形状为 (batch_size, 1)，表示填充点的标签
            points = torch.cat([points, padding_point], dim=1)  # 在点坐标张量的第二个维度上拼接填充点
            labels = torch.cat([labels, padding_label], dim=1)  # 在标签张量的第二个维度上拼接填充标签
        point_embedding = self.pe_layer.forward_with_coords(points,
                                                            self.input_image_size)  # 调用位置编码层的 forward_with_coords 方法，根据点坐标和输入图像大小生成点的位置编码
        point_embedding[labels == -1] = 0.0  # 将标签为 -1 的点的嵌入设置为全零
        point_embedding[
            labels == -1] += self.not_a_point_embed.weight  # 为标签为 -1 的点的嵌入添加 not_a_point_embed 层的权重，以表示这不是一个有效的点
        point_embedding[labels == 0] += self.point_embeddings[0].weight  # 为标签为 0 的点的嵌入添加第一个点嵌入层的权重
        point_embedding[labels == 1] += self.point_embeddings[1].weight  # 为标签为 1 的点的嵌入添加第二个点嵌入层的权重
        return point_embedding

    def _embed_boxes(self,
                     boxes: torch.Tensor) -> torch.Tensor:  # 输入的边界框张量，形状通常为 (batch_size, 4)，其中 4 表示边界框的 (x1, y1, x2, y2) 坐标
        """Embeds box prompts."""  # 对输入的边界框提示进行嵌入编码
        boxes = boxes + 0.5  # Shift to center of pixel # 将边界框坐标偏移 0.5，使其移动到像素的中心位置
        coords = boxes.reshape(-1, 2, 2)  # 将边界框张量重塑为 (batch_size, 2, 2) 的形状，其中第二个维度的 2 表示边界框的两个角点
        corner_embedding = self.pe_layer.forward_with_coords(coords,
                                                             self.input_image_size)  # 调用位置编码层的 forward_with_coords 方法，根据边界框角点坐标和输入图像大小生成角点的位置编码
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight  # 为边界框的第一个角点的嵌入添加第三个点嵌入层的权重
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight  # 为边界框的第二个角点的嵌入添加第四个点嵌入层的权重
        return corner_embedding

    def _embed_rboxes(self,
                      rboxes: torch.Tensor) -> torch.Tensor:  # 输入的边界框张量，形状通常为 (batch_size, 4, 2)，其中 4 表示边界框的 (x1, y1, x2, y2) 坐标
        """Embeds rbox prompts."""  # 对输入的边界框提示进行嵌入编码
        boxes = rboxes + 0.5  # Shift to center of pixel # 将边界框坐标偏移 0.5，使其移动到像素的中心位置
        coords = boxes.reshape(-1, 4, 2)  # 将边界框张量重塑为 (batch_size, 4, 2) 的形状，其中第二个维度的 4 表示边界框的4个角点
        rbox_embedding = self.pe_layer.forward_with_coords(coords,
                                                           self.input_image_size)  # 调用位置编码层的 forward_with_coords 方法，根据边界框角点坐标和输入图像大小生成角点的位置编码  b,4,dim
        rbox_embedding[:, 0, :] += self.rbox_point_embeddings[0].weight  # 为边界框的第一个角点的嵌入添加第三个点嵌入层的权重
        rbox_embedding[:, 1, :] += self.rbox_point_embeddings[1].weight  # 为边界框的第二个角点的嵌入添加第四个点嵌入层的权重
        rbox_embedding[:, 2, :] += self.rbox_point_embeddings[2].weight  # 为边界框的第一个角点的嵌入添加第三个点嵌入层的权重
        rbox_embedding[:, 3, :] += self.rbox_point_embeddings[3].weight  # 为边界框的第二个角点的嵌入添加第四个点嵌入层的权重
        return rbox_embedding

    def _embed_bboxes_origin(self,
                             bboxes: torch.Tensor) -> torch.Tensor:  # 输入的边界框张量，形状通常为 (batch_size, 4, 2)，其中 4 表示边界框的 (x1, y1, x2, y2) 坐标
        """Embeds rbox prompts."""  # 对输入的边界框提示进行嵌入编码
        boxes = bboxes + 0.5  # Shift to center of pixel # 将边界框坐标偏移 0.5，使其移动到像素的中心位置
        coords = boxes.reshape(-1, 2, 2)  # 将边界框张量重塑为 (batch_size, 4, 2) 的形状，其中第二个维度的 4 表示边界框的4个角点
        bbox_embedding_origin = self.pe_layer.forward_with_coords(coords,
                                                                  self.input_image_size)  # 调用位置编码层的 forward_with_coords 方法，根据边界框角点坐标和输入图像大小生成角点的位置编码  b,2,dim
        bbox_embedding_origin[:, 0, :] += self.bbox_point_embeddings_origin[0].weight  # 为边界框的第一个角点的嵌入添加第三个点嵌入层的权重
        bbox_embedding_origin[:, 1, :] += self.bbox_point_embeddings_origin[1].weight  # 为边界框的第二个角点的嵌入添加第四个点嵌入层的权重
        return bbox_embedding_origin

    def _embed_rboxes_origin(self,
                             rboxes: torch.Tensor) -> torch.Tensor:  # 输入的边界框张量，形状通常为 (batch_size, 4, 2)，其中 4 表示边界框的 (x1, y1, x2, y2) 坐标
        """Embeds rbox prompts."""  # 对输入的边界框提示进行嵌入编码
        boxes = rboxes + 0.5  # Shift to center of pixel # 将边界框坐标偏移 0.5，使其移动到像素的中心位置
        coords = boxes.reshape(-1, 4, 2)  # 将边界框张量重塑为 (batch_size, 4, 2) 的形状，其中第二个维度的 4 表示边界框的4个角点
        rbox_embedding_origin = self.pe_layer.forward_with_coords(coords,
                                                                  self.input_image_size)  # 调用位置编码层的 forward_with_coords 方法，根据边界框角点坐标和输入图像大小生成角点的位置编码  b,4,dim
        rbox_embedding_origin[:, 0, :] += self.rbox_point_embeddings_origin[0].weight  # 为边界框的第一个角点的嵌入添加第三个点嵌入层的权重
        rbox_embedding_origin[:, 1, :] += self.rbox_point_embeddings_origin[1].weight  # 为边界框的第二个角点的嵌入添加第四个点嵌入层的权重
        rbox_embedding_origin[:, 2, :] += self.rbox_point_embeddings_origin[2].weight  # 为边界框的第一个角点的嵌入添加第三个点嵌入层的权重
        rbox_embedding_origin[:, 3, :] += self.rbox_point_embeddings_origin[3].weight  # 为边界框的第二个角点的嵌入添加第四个点嵌入层的权重
        return rbox_embedding_origin

    def _embed_masks(self,
                     masks: torch.Tensor) -> torch.Tensor:  # 输入的掩码张量，形状通常为 (batch_size, 1, H, W)，其中 H 和 W 是掩码的高度和宽度
        """Embeds mask inputs."""  # 对输入的掩码进行嵌入编码
        mask_embedding = self.mask_downscaling(masks)  # 将掩码张量输入到掩码下采样模块中进行下采样和特征转换
        return mask_embedding

    def _embed_masks_rbox_origin(self,
                                 rboxes: torch.Tensor) -> torch.Tensor:  # 输入的掩码张量，形状通常为 (batch_size, 1, H, W)，其中 H 和 W 是掩码的高度和宽度
        transform = ResizeLongestSide(self.image_embedding_size[0])
        rboxes = torch.tensor(transform.apply_rboxes(rboxes.cpu().numpy(), self.input_image_size),
                              device=self._get_device())  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配
        """Embeds mask inputs."""  # 对输入的掩码进行嵌入编码
        # rboxes = rboxes + 0.5  # Shift to center of pixel # 将边界框坐标偏移 0.5，使其移动到像素的中心位置
        # rboxes = rboxes.reshape(-1, 4, 2)  # 将边界框张量重塑为 (batch_size, 4, 2) 的形状，其中第二个维度的 4 表示边界框的4个角点

        b = rboxes.shape[0]
        h, w = self.image_embedding_size[0], self.image_embedding_size[1]
        c = self.embed_dim
        results = []

        # 获取嵌入向量
        embed = self.rbox_mask_embed.weight.reshape(-1, 1)

        for i in range(b):
            # 将当前 rboxes_origin 转换为 numpy 数组并转为整数类型
            current_rbox = rboxes[i].cpu().numpy().astype(np.int32)
            # 调整形状以符合 cv2.fillPoly 的要求
            # current_rbox = current_rbox.reshape(-1, 1, 2)

            # 创建一个掩码，用于标记四点区域
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [current_rbox], 255)
            mask = torch.from_numpy(mask).bool().to(rboxes.device)
            # 找出 mask 中为 True 的位置的坐标
            y_coords, x_coords = torch.where(mask)
            num_true = y_coords.size(0)
            # print(i)
            embed_new = embed.expand(-1, num_true)

            # 创建当前批次的空白图像
            current_image = torch.zeros((c, h, w), dtype=torch.float32).to(rboxes.device)
            # 赋值操作
            current_image[:, y_coords, x_coords] = embed_new

            results.append(current_image)

            # 将所有批次按批次维度堆叠
        result_images = torch.stack(results, dim=0)

        return result_images

    def _get_batch_size(  # 据输入提示的批量大小来确定输出的批量大小
            self,
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],  # 点提示，包含点坐标和对应的标签，可能为 None
            boxes: Optional[torch.Tensor],  # 边界框提示，可能为 None
            boxes_origin: Optional[torch.Tensor],  # 边界框提示，可能为 None
            rboxes: Optional[torch.Tensor],  # 边界框提示，可能为 None
            rboxes_origin: Optional[torch.Tensor],  # 边界框提示，可能为 None
            masks: Optional[torch.Tensor],  # 掩码提示，可能为 None
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]  # # 如果存在点提示，则返回点提示的批量大小
        elif boxes is not None:
            return boxes.shape[0]  # 如果存在边界框提示，则返回边界框提示的批量大小
        elif boxes_origin is not None:
            return boxes_origin.shape[0]  # 如果存在边界框提示，则返回边界框提示的批量大小
        elif rboxes is not None:
            return rboxes.shape[0]  # 如果存在边界框提示，则返回边界框提示的批量大小
        elif rboxes_origin is not None:
            return rboxes_origin.shape[0]  # 如果存在边界框提示，则返回边界框提示的批量大小
        elif masks is not None:
            return masks.shape[0]  # 如果存在掩码提示，则返回掩码提示的批量大小
        else:
            return 1  # 如果所有提示都不存在，则默认批量大小为 1

    def _get_device(self) -> torch.device:  # 获取模型参数所在的设备（如 CPU 或 GPU）
        return self.point_embeddings[0].weight.device  # 通过获取第一个点嵌入层的权重所在的设备来确定整个模型的设备

    def forward(
            self,
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],  # 点提示，包含点坐标和对应的标签，可能为 None
            boxes: Optional[torch.Tensor],  # 边界框提示，可能为 None
            boxes_origin: Optional[torch.Tensor],  # 边界框提示，可能为 None
            rboxes: Optional[torch.Tensor],  # 边界框提示，可能为 None
            rboxes_origin: Optional[torch.Tensor],  # 边界框提示，可能为 None
            masks: Optional[torch.Tensor],  # 掩码提示，可能为 None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.#对不同类型的提示进行嵌入编码，返回稀疏嵌入和密集嵌入

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.#点和边界框的稀疏嵌入，形状为 BxNx(embed_dim)，其中 B 是批量大小，N 由输入的点和边界框数量决定
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)#掩码的密集嵌入，形状为 Bx(embed_dim)x(embed_H)x(embed_W)，其中 embed_H 和 embed_W 是图像嵌入的高度和宽度
        """
        bs = self._get_batch_size(points, boxes, boxes_origin, rboxes, rboxes_origin,
                                  masks)  # 调用 _get_batch_size 方法获取批量大小
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim),
                                        device=self._get_device())  # 初始化稀疏嵌入张量，形状为 (bs, 0, self.embed_dim)，并放置在模型所在的设备上
        if points is not None:  # 如果存在点提示
            coords, labels = points  # 解包点提示，得到点坐标和标签
            point_embeddings = self._embed_points(coords, labels, pad=(
                        boxes is None))  # 调用 _embed_points 方法对这些点进行嵌入编码，根据是否存在边界框决定是否进行填充
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings],
                                          dim=1)  # 将点的嵌入结果沿着第二个维度（即嵌入数量维度）拼接到稀疏嵌入张量中
        if boxes is not None:  # 如果存在边界框提示
            box_embeddings = self._embed_boxes(boxes)  # 调用 _embed_boxes 方法对边界框进行嵌入编码
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)  # 将边界框的嵌入结果沿着第二个维度拼接到稀疏嵌入张量中
        if boxes_origin is not None:  # 如果存在边界框提示
            box_embeddings_origin = self._embed_bboxes_origin(boxes_origin)  # 调用 _embed_boxes 方法对边界框进行嵌入编码
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings_origin],
                                          dim=1)  # 将边界框的嵌入结果沿着第二个维度拼接到稀疏嵌入张量中
        if rboxes is not None:  # 如果存在边界框提示
            rbox_embeddings = self._embed_rboxes(rboxes)  # 调用 _embed_boxes 方法对边界框进行嵌入编码
            sparse_embeddings = torch.cat([sparse_embeddings, rbox_embeddings], dim=1)  # 将边界框的嵌入结果沿着第二个维度拼接到稀疏嵌入张量中
        if rboxes_origin is not None:  # 如果存在边界框提示
            rbox_embeddings_origin = self._embed_rboxes_origin(rboxes_origin)  # 调用 _embed_boxes 方法对边界框进行嵌入编码
            sparse_embeddings = torch.cat([sparse_embeddings, rbox_embeddings_origin],
                                          dim=1)  # 将边界框的嵌入结果沿着第二个维度拼接到稀疏嵌入张量中

        if masks is not None:  # 如果存在掩码提示
            dense_embeddings = self._embed_masks(masks)  # 调用 _embed_masks 方法对掩码进行嵌入编码
        else:  # 如果不存在掩码提示
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )  # 使用 no_mask_embed 层的权重进行重塑和扩展，生成默认的密集嵌入
        # if rboxes is not None:
        # dense_embeddings = dense_embeddings+self._embed_masks_rbox_origin(rboxes)

        return sparse_embeddings, dense_embeddings  # 返回稀疏嵌入和密集嵌入


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.#使用随机空间频率进行位置编码
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        # num_pos_feats (int): 位置编码特征的数量，默认为 64
        # scale (Optional[float]): 缩放因子，用于生成随机高斯矩阵，若为 None 或小于等于 0，则默认设为 1.0
        super().__init__()
        if scale is None or scale <= 0.0:  # 如果 scale 为 None 或者小于等于 0，将其设置为 1.0
            scale = 1.0
        self.register_buffer(  # 注册一个缓冲区，用于存储随机生成的高斯矩阵
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),  # 该矩阵形状为 (2, num_pos_feats)，用于后续的位置编码计算
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        # coords (torch.Tensor): 归一化后的坐标张量，形状为 d_1 x ... x d_n x 2
        """Positionally encode points that are normalized to [0,1]."""  # 对归一化到 [0, 1] 范围内的点进行位置编码
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1  # 假设输入的坐标在 [0, 1]^2 正方形内，将坐标从 [0, 1] 范围转换到 [-1, 1] 范围
        coords = coords @ self.positional_encoding_gaussian_matrix  # 将坐标与高斯矩阵相乘，得到新的坐标表示
        coords = 2 * np.pi * coords  # 将坐标乘以 2π，用于后续的正弦和余弦计算
        # outputs d_1 x ... x d_n x C shape
        # 对坐标分别取正弦和余弦值，并在最后一个维度上拼接
        # 最终输出形状为 d_1 x ... x d_n x C 的位置编码张量
        return torch.cat([torch.sin(coords), torch.cos(coords)],
                         dim=-1)  # torch.Tensor: 位置编码后的张量，形状为 d_1 x ... x d_n x C

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        # size(Tuple[int, int]): 网格的大小，格式为(高度, 宽度)
        """Generate positional encoding for a grid of the specified size."""  # 为指定大小的网格生成位置编码
        h, w = size  # 解包网格的高度和宽度
        device: Any = self.positional_encoding_gaussian_matrix.device  # 获取高斯矩阵所在的设备，确保后续操作在同一设备上进行
        grid = torch.ones((h, w), device=device, dtype=torch.float32)  # 创建一个全为 1 的张量，形状为 (h, w)，用于后续的坐标生成
        y_embed = grid.cumsum(dim=0) - 0.5  # 在第 0 维上进行累加求和，并减去 0.5，得到 y 方向的嵌入坐标
        x_embed = grid.cumsum(dim=1) - 0.5  # 在第 1 维上进行累加求和，并减去 0.5，得到 x 方向的嵌入坐标
        y_embed = y_embed / h  # 将 y 方向的嵌入坐标除以高度 h，进行归一化
        x_embed = x_embed / w  # 将 x 方向的嵌入坐标除以宽度 w，进行归一化
        # 将 x 和 y 方向的嵌入坐标在最后一个维度上堆叠
        # 调用 _pe_encoding 方法进行位置编码
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        # 调整位置编码张量的维度顺序，从 H x W x C 变为 C x H x W
        # torch.Tensor: 位置编码张量，形状为 C x H x W
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""  # 对未归一化到 [0, 1] 范围内的点进行位置编码
        coords = coords_input.clone()  # 复制输入的坐标，避免修改原始数据
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]  # 将 x 坐标除以图像的宽度，进行归一化
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]  # 将 y 坐标除以图像的高度，进行归一化
        # 将坐标转换为浮点类型，并调用 _pe_encoding 方法进行位置编码
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
