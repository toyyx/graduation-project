# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int, # 变压器（Transformer）模型的通道维度，用于确定特征的表示维度
        transformer: nn.Module, # 用于预测掩码（masks）的变压器（Transformer）模型
        num_multimask_outputs: int = 3, # 在消除掩码歧义时需要预测的掩码数量，默认为 3
        activation: Type[nn.Module] = nn.GELU, # 上采样掩码时使用的激活函数类型，默认为 GELU
        iou_head_depth: int = 3, # 用于预测掩码质量的多层感知机（MLP）的深度，默认为 3
        iou_head_hidden_dim: int = 256,# 用于预测掩码质量的多层感知机（MLP）的隐藏层维度，默认为 256
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.#使用变压器（Transformer）架构，根据图像和提示嵌入预测掩码。

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim# 保存变压器（Transformer）模型的通道维度
        self.transformer = transformer # 保存用于预测掩码的变压器（Transformer）模型

        self.num_multimask_outputs = num_multimask_outputs # 保存消除掩码歧义时需要预测的掩码数量

        self.iou_token = nn.Embedding(1, transformer_dim)# 创建一个嵌入层，用于生成 IoU（交并比）标记的嵌入向量
        self.num_mask_tokens = num_multimask_outputs + 1  # 计算掩码标记的总数，包括用于预测 IoU 的标记
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)  # 创建一个嵌入层，用于生成掩码标记的嵌入向量

        self.output_upscaling = nn.Sequential( # 定义一个顺序容器，用于对输出进行上采样操作
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),  # 第一个反卷积层，将通道数从 transformer_dim 减少到 transformer_dim // 4
            LayerNorm2d(transformer_dim // 4), # 二维层归一化层，对通道维度进行归一化
            activation(), # 应用指定的激活函数
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2), # 第二个反卷积层，将通道数从 transformer_dim // 4 减少到 transformer_dim // 8
            activation(), # 再次应用指定的激活函数
        )
        self.output_hypernetworks_mlps = nn.ModuleList(# 创建一个模块列表，包含多个多层感知机（MLP），每个 MLP 用于生成一个掩码的超网络参数
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP( # 创建一个多层感知机（MLP），用于预测掩码的质量（IoU）
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,  # 图像编码器输出的嵌入向量
        image_pe: torch.Tensor,# 与图像嵌入向量形状相同的位置编码
        sparse_prompt_embeddings: torch.Tensor, # 点和框的嵌入向量
        dense_prompt_embeddings: torch.Tensor, # 掩码输入的嵌入向量
        multimask_output: bool,# 是否返回多个掩码还是单个掩码的布尔值
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.#根据图像和提示嵌入预测掩码。

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks #批量预测的掩码
          torch.Tensor: batched predictions of mask quality #批量预测的掩码质量
        """
        masks, iou_pred = self.predict_masks( # 调用 predict_masks 方法进行掩码预测，并得到预测的掩码和掩码质量预测值
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output # 根据 multimask_output 的值选择要输出的掩码
        if multimask_output:
            mask_slice = slice(1, None) # 如果需要返回多个掩码，选择除第一个掩码之外的所有掩码
        else:
            mask_slice = slice(0, 1)# 如果只需要返回单个掩码，选择第一个掩码
        masks = masks[:, mask_slice, :, :] # 根据选择的切片操作选择掩码
        iou_pred = iou_pred[:, mask_slice]# 根据选择的切片操作选择掩码质量预测值

        # Prepare output
        return masks, iou_pred # 准备输出结果

    def predict_masks(
        self,
        image_embeddings: torch.Tensor, # 图像编码器输出的嵌入向量，形状通常为 (batch_size, channels, height, width)
        image_pe: torch.Tensor,  # 与图像嵌入向量形状相同的位置编码，用于为图像特征添加位置信息
        sparse_prompt_embeddings: torch.Tensor,# 稀疏提示的嵌入向量，例如点和框的嵌入，形状为 (batch_size, num_sparse_prompts, dim)
        dense_prompt_embeddings: torch.Tensor, # 密集提示的嵌入向量，例如掩码输入的嵌入，形状通常与图像嵌入向量相关，可用于增强图像特征
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""  # 此方法用于预测掩码，具体细节可参考 forward 方法
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0) # 将 IoU 标记的权重和掩码标记的权重在第 0 维（特征维度）上进行拼接
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1) # 为 output_tokens 添加一个批次维度（第 0 维），并将其扩展为与 sparse_prompt_embeddings 相同的批次大小
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  # 将扩展后的 output_tokens 与 sparse_prompt_embeddings 在第 1 维（序列长度维度）上进行拼接

        # Expand per-image data in batch direction to be per-mask
        # 在批次方向上扩展每个图像的数据，使其对应每个掩码
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) # 将 image_embeddings 在批次维度上重复 tokens 的批次大小次，确保每个掩码都有对应的图像特征
        src = src + dense_prompt_embeddings # 将扩展后的图像嵌入与密集提示嵌入相加，融合两者的信息
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)  # 同样地，将图像位置编码在批次维度上重复 tokens 的批次大小次
        b, c, h, w = src.shape  # 获取融合后特征的批次大小、通道数、高度和宽度

        # Run the transformer  # 运行 Transformer 模型
        hs, src = self.transformer(src, pos_src, tokens) # 将处理后的特征 src、位置编码 pos_src 和拼接后的标记 tokens 输入到 Transformer 模型中
        iou_token_out = hs[:, 0, :] # 从 Transformer 输出中提取 IoU 标记对应的输出
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :] # 从 Transformer 输出中提取掩码标记对应的输出

        # Upscale mask embeddings and predict masks using the mask tokens
        # 对掩码嵌入进行上采样，并使用掩码标记预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)# 调整 src 的维度顺序，使其适合后续处理
        upscaled_embedding = self.output_upscaling(src)# 使用定义好的上采样模块对 src 进行上采样操作#b32 256 256
        hyper_in_list: List[torch.Tensor] = [] # 初始化一个列表，用于存储每个掩码标记经过超网络多层感知机处理后的输出
        for i in range(self.num_mask_tokens):  # 遍历每个掩码标记
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])) # 将每个掩码标记的输出输入到对应的超网络多层感知机中，并将结果添加到列表中
        hyper_in = torch.stack(hyper_in_list, dim=1) # 将列表中的元素在第 1 维上堆叠成一个张量
        b, c, h, w = upscaled_embedding.shape # 获取上采样后嵌入的批次大小、通道数、高度和宽度
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w) # 通过矩阵乘法将超网络的输出与上采样后的嵌入进行结合，得到掩码预测结果

        # Generate mask quality predictions
        # 生成掩码质量预测结果
        iou_pred = self.iou_prediction_head(iou_token_out)  # 将 IoU 标记的输出输入到 IoU 预测头中，得到掩码质量的预测值

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def  __init__(
        self,
        input_dim: int,# 输入特征的维度，即输入到多层感知机（MLP）的向量的长度
        hidden_dim: int,  # 隐藏层的维度，每个隐藏层的神经元数量
        output_dim: int, # 输出特征的维度，即 MLP 最终输出的向量的长度
        num_layers: int,# MLP 的层数，包括输入层、隐藏层和输出层
        sigmoid_output: bool = False, # 是否在输出层使用 Sigmoid 激活函数，默认为 False
    ) -> None:
        super().__init__()
        self.num_layers = num_layers # 保存 MLP 的层数
        h = [hidden_dim] * (num_layers - 1) # 创建一个列表 h，其中包含 num_layers - 1 个 hidden_dim，用于表示隐藏层的维度
        self.layers = nn.ModuleList( # 创建一个 nn.ModuleList，其中包含多个 nn.Linear 层
            # 这些线性层的输入和输出维度由 [input_dim] + h 和 h + [output_dim] 确定
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output  # 保存是否在输出层使用 Sigmoid 激活函数的标志

    def forward(self, x):
        for i, layer in enumerate(self.layers): # 遍历 MLP 的每一层
            # 如果当前层不是最后一层，则使用 ReLU 激活函数对线性层的输出进行激活
            # 如果是最后一层，则不使用激活函数，直接输出线性层的结果
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:# 如果 sigmoid_output 为 True，则在输出层使用 Sigmoid 激活函数
            x = F.sigmoid(x)
        return x # 返回 MLP 的最终输出
