# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int, # 变压器（Transformer）中的层数，决定了模型的复杂度和特征提取能力
        embedding_dim: int, # 输入嵌入的通道维度，即每个嵌入向量的长度
        num_heads: int, # 多头注意力机制中的头数，必须能整除 embedding_dim
        mlp_dim: int,  # 多层感知机（MLP）块内部的通道维度
        activation: Type[nn.Module] = nn.ReLU, # MLP 块中使用的激活函数类型，默认为 ReLU
        attention_downsample_rate: int = 2,  # 注意力机制中的下采样率，用于减少计算量
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.
        # 一个变压器（Transformer）解码器，使用提供位置嵌入的查询来关注输入图像。

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth # 保存变压器（Transformer）的层数
        self.embedding_dim = embedding_dim# 保存输入嵌入的通道维度
        self.num_heads = num_heads # 保存多头注意力机制中的头数
        self.mlp_dim = mlp_dim # 保存多层感知机（MLP）块内部的通道维度
        self.layers = nn.ModuleList()# 创建一个 nn.ModuleList 用于存储变压器（Transformer）的每一层

        for i in range(depth): # 遍历每一层
            self.layers.append( # 向 layers 列表中添加一个 TwoWayAttentionBlock 层
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,# 输入嵌入的通道维度
                    num_heads=num_heads,  # 多头注意力机制中的头数
                    mlp_dim=mlp_dim,# 多层感知机（MLP）块内部的通道维度
                    activation=activation, # MLP 块中使用的激活函数
                    attention_downsample_rate=attention_downsample_rate,# 注意力机制中的下采样率
                    skip_first_layer_pe=(i == 0),  # 是否跳过第一层的位置编码，只有第一层设置为 True
                )
            )

        self.final_attn_token_to_image = Attention( # 创建一个 Attention 层，用于从点嵌入到图像嵌入的最终注意力计算
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)# 创建一个 LayerNorm 层，用于对最终的查询嵌入进行归一化

    def forward(
        self,
        image_embedding: Tensor, # 要关注的图像嵌入，形状为 B x embedding_dim x h x w，B 是批量大小，h 和 w 是图像的高度和宽度
        image_pe: Tensor, # 要添加到图像嵌入的位置编码，必须与 image_embedding 形状相同
        point_embedding: Tensor, # 要添加到查询点的嵌入，形状为 B x N_points x embedding_dim，N_points 是查询点的数量
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding  #处理后的点嵌入
          torch.Tensor: the processed image_embedding  #处理后的图像嵌入
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape# 获取图像嵌入的批量大小、通道数、高度和宽度
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # 将图像嵌入的高度和宽度维度展平，并调整维度顺序为 B x N_image_tokens x C
        image_pe = image_pe.flatten(2).permute(0, 2, 1) # 将图像位置编码的高度和宽度维度展平，并调整维度顺序为 B x N_image_tokens x C

        # Prepare queries
        queries = point_embedding # 将点嵌入作为查询
        keys = image_embedding  # 将图像嵌入作为键

        # Apply transformer blocks and final layernorm
        # 应用变压器（Transformer）块和最终的层归一化
        for layer in self.layers:# 遍历变压器（Transformer）的每一层
            queries, keys = layer(# 将查询、键、查询位置编码和键位置编码输入到每一层进行处理
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        # 应用从点嵌入到图像嵌入的最终注意力层
        q = queries + point_embedding # 将查询和查询位置编码相加，得到最终的查询
        k = keys + image_pe # 将键和键位置编码相加，得到最终的键
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys) # 计算最终的注意力输出
        queries = queries + attn_out# 将最终的注意力输出添加到查询中
        queries = self.norm_final_attn(queries)# 对查询进行层归一化

        return queries, keys# 返回处理后的点嵌入和图像嵌入


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int, # 嵌入向量的通道维度，也就是每个嵌入向量的长度
        num_heads: int, # 注意力层中多头注意力机制的头数
        mlp_dim: int = 2048, # 多层感知机（MLP）块的隐藏层维度，默认为 2048
        activation: Type[nn.Module] = nn.ReLU,# MLP 块中使用的激活函数类型，默认为 ReLU
        attention_downsample_rate: int = 2, # 注意力层中的下采样率，用于减少计算量，默认为 2
        skip_first_layer_pe: bool = False, # 是否跳过第一层的位置编码，默认为 False
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.
         一个包含四层的变压器（Transformer）块：
        （1）稀疏输入的自注意力层；
        （2）稀疏输入对密集输入的交叉注意力层；
        （3）对稀疏输入进行处理的 MLP 块；
        （4）密集输入对稀疏输入的交叉注意力层。

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads) # 创建一个自注意力层，用于处理稀疏输入
        self.norm1 = nn.LayerNorm(embedding_dim) # 创建一个层归一化层，用于对自注意力层的输出进行归一化

        self.cross_attn_token_to_image = Attention( # 创建一个交叉注意力层，用于稀疏输入对密集输入的注意力计算
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim) # 创建一个层归一化层，用于对该交叉注意力层的输出进行归一化

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation) # 创建一个 MLP 块，用于对稀疏输入进行非线性变换
        self.norm3 = nn.LayerNorm(embedding_dim) # 创建一个层归一化层，用于对 MLP 块的输出进行归一化

        self.norm4 = nn.LayerNorm(embedding_dim) # 创建一个层归一化层，用于对密集输入对稀疏输入的交叉注意力层的输入进行归一化
        self.cross_attn_image_to_token = Attention(  # 创建一个交叉注意力层，用于密集输入对稀疏输入的注意力计算
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe # 保存是否跳过第一层位置编码的标志

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # queries 是稀疏输入，通常代表点嵌入
        # keys 是密集输入，通常代表图像嵌入
        # query_pe 是查询的位置编码
        # key_pe 是键的位置编码

        # Self attention block
        # 自注意力层，处理稀疏输入
        if self.skip_first_layer_pe: # 如果跳过第一层位置编码，直接对查询进行自注意力计算
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe # 将查询和查询位置编码相加，得到带有位置信息的查询
            attn_out = self.self_attn(q=q, k=q, v=queries) # 进行自注意力计算，得到注意力输出
            queries = queries + attn_out# 将注意力输出添加到原始查询上，更新查询
        queries = self.norm1(queries) # 对更新后的查询进行层归一化

        # Cross attention block, tokens attending to image embedding
        # 交叉注意力层，稀疏输入（查询）关注密集输入（键）
        q = queries + query_pe # 将查询和查询位置编码相加，得到带有位置信息的查询
        k = keys + key_pe# 将键和键位置编码相加，得到带有位置信息的键
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys) # 进行交叉注意力计算，查询关注键，得到注意力输出
        queries = queries + attn_out # 将注意力输出添加到原始查询上，更新查询
        queries = self.norm2(queries) # 对更新后的查询进行层归一化

        # MLP block
        # MLP 块，对稀疏输入进行非线性变换
        mlp_out = self.mlp(queries) # 将查询输入到 MLP 块中，得到 MLP 输出
        queries = queries + mlp_out# 将 MLP 输出添加到原始查询上，更新查询
        queries = self.norm3(queries)# 对更新后的查询进行层归一化

        # Cross attention block, image embedding attending to tokens
        # 交叉注意力层，密集输入（键）关注稀疏输入（查询）
        q = queries + query_pe # 将查询和查询位置编码相加，得到带有位置信息的查询
        k = keys + key_pe  # 将键和键位置编码相加，得到带有位置信息的键
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries) # 进行交叉注意力计算，键关注查询，得到注意力输出
        keys = keys + attn_out # 将注意力输出添加到原始键上，更新键
        keys = self.norm4(keys)   # 对更新后的键进行层归一化

        return queries, keys# 返回更新后的查询和键


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
     #一个注意力层，支持在将输入投影为查询（queries）、键（keys）和值（values）后，对嵌入向量的大小进行下采样。
    """

    def __init__(
        self,
        embedding_dim: int,  # 输入嵌入向量的维度
        num_heads: int, # 多头注意力机制中的头数
        downsample_rate: int = 1,# 下采样率，用于控制投影后嵌入向量维度的缩减比例，默认为 1 即不进行下采样
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim# 保存输入嵌入向量的维度
        self.internal_dim = embedding_dim // downsample_rate # 计算投影后内部嵌入向量的维度，通过输入维度除以下采样率得到
        self.num_heads = num_heads # 保存多头注意力机制的头数
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim." # 确保内部维度能被头数整除，否则抛出异常

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim) # 定义查询的投影层，将输入嵌入向量投影到内部维度
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim) # 定义键的投影层，将输入嵌入向量投影到内部维度
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)# 定义值的投影层，将输入嵌入向量投影到内部维度
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)# 定义输出的投影层，将内部维度的向量投影回输入嵌入向量的维度

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:# 此方法用于将输入张量按头数进行分割，实现多头注意力机制
        b, n, c = x.shape # 获取输入张量的批次大小、序列长度和通道数
        x = x.reshape(b, n, num_heads, c // num_heads) # 重塑输入张量，将通道维度分割为头数和每个头的通道数
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head  # 交换第 1 维和第 2 维，使得头数维度提前

    def _recombine_heads(self, x: Tensor) -> Tensor:# 此方法用于将按头分割的张量重新组合成一个张量
        b, n_heads, n_tokens, c_per_head = x.shape  # 获取输入张量的批次大小、头数、序列长度和每个头的通道数
        x = x.transpose(1, 2)  # 交换第 1 维和第 2 维，恢复原来的维度顺序
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C  # 重塑张量，将头数和每个头的通道数合并为一个通道维度

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor: # 前向传播方法，计算注意力输出
        # Input projections
        # 对输入的查询、键和值进行投影，将其维度转换为内部维度
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        # 将投影后的查询、键和值按头数进行分割，实现多头注意力机制
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape # 获取每个头的通道数
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens  # 计算查询和键的点积，得到注意力分数矩阵
        attn = attn / math.sqrt(c_per_head)  # 对注意力分数矩阵进行缩放，除以每个头通道数的平方根，防止点积结果过大
        attn = torch.softmax(attn, dim=-1) # 对注意力分数矩阵在最后一个维度上进行 softmax 操作，得到注意力权重

        # Get output
        out = attn @ v # 根据注意力权重对值进行加权求和，得到每个头的输出
        out = self._recombine_heads(out) # 将每个头的输出重新组合成一个张量
        out = self.out_proj(out)# 通过输出投影层，将输出张量的维度转换回输入嵌入向量的维度

        return out# 返回最终的注意力输出
