# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_sam_vit_h(checkpoint=None):# 构建基于 Vision Transformer (ViT) 架构的 SAM（Segment Anything Model）的大尺寸（Huge）版本。
    # checkpoint (str, 可选): 预训练模型的检查点文件路径。如果提供，模型将从该检查点加载预训练权重；如果为 None，则使用随机初始化的权重。
    return _build_sam(# SAM 模型实例: 一个已经配置好的 SAM 模型实例，使用 ViT-H 架构。
        encoder_embed_dim=1280,# 编码器嵌入维度，即模型中特征向量的维度大小，这里设置为 1280
        encoder_depth=32,# 编码器的深度，即 Transformer 块的数量，这里设置为 32
        encoder_num_heads=16, # 编码器中多头注意力机制的头数，这里设置为 16
        encoder_global_attn_indexes=[7, 15, 23, 31],# 编码器中使用全局注意力机制的 Transformer 块的索引列表，这些块在处理长距离依赖时更有效
        checkpoint=checkpoint, # 预训练模型的检查点文件路径
    )


build_sam = build_sam_vit_h # 将 build_sam 别名指向 build_sam_vit_h，意味着默认构建的是 ViT-H 版本的 SAM 模型


def build_sam_vit_l(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,  # 编码器嵌入维度，这里设置为 1024
        encoder_depth=24,  # 编码器的深度，这里设置为 24
        encoder_num_heads=16, # 编码器中多头注意力机制的头数，这里设置为 16
        encoder_global_attn_indexes=[5, 11, 17, 23], # 编码器中使用全局注意力机制的 Transformer 块的索引列表
        checkpoint=checkpoint,# 预训练模型的检查点文件路径
    )


def build_sam_vit_b(checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768, # 编码器嵌入维度，这里设置为 768
        encoder_depth=12, # 编码器的深度，这里设置为 12
        encoder_num_heads=12,  # 编码器中多头注意力机制的头数，这里设置为 12
        encoder_global_attn_indexes=[2, 5, 8, 11], # 编码器中使用全局注意力机制的 Transformer 块的索引列表
        checkpoint=checkpoint,# 预训练模型的检查点文件路径
    )


sam_model_registry = {# 定义一个字典，将不同的模型名称映射到对应的模型构建函数
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim, # 图像编码器的嵌入维度，决定了编码器输出特征的维度大小
    encoder_depth, # 图像编码器中 Transformer 块的数量，体现了编码器的深度和复杂度
    encoder_num_heads,# 图像编码器中多头注意力机制的头数
    encoder_global_attn_indexes,# 图像编码器中使用全局注意力机制的 Transformer 块的索引列表
    checkpoint=None,# 预训练模型的检查点文件路径，如果为 None，则使用随机初始化的权重
):
    prompt_embed_dim = 256 # 提示编码器和掩码解码器中使用的嵌入维度
    image_size = 1024 # 输入图像的尺寸，这里固定为 1024x1024
    vit_patch_size = 16# Vision Transformer（ViT）中每个 patch 的大小
    image_embedding_size = image_size // vit_patch_size # 图像嵌入的尺寸，通过输入图像尺寸除以 patch 大小得到
    sam = Sam( # 创建 SAM 模型实例
        image_encoder=ImageEncoderViT( # 初始化图像编码器
            depth=encoder_depth,# Transformer 块的数量
            embed_dim=encoder_embed_dim,# 嵌入维度
            img_size=image_size,# 输入图像的尺寸
            mlp_ratio=4,# MLP 层中隐藏层维度与嵌入维度的比例
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), # 归一化层，使用 torch.nn.LayerNorm 并设置 eps 为 1e-6
            num_heads=encoder_num_heads,# 多头注意力机制的头数
            patch_size=vit_patch_size,# patch 的大小
            qkv_bias=True,  # 是否使用偏置项在 QKV 投影中
            use_rel_pos=True,# 是否使用相对位置编码
            global_attn_indexes=encoder_global_attn_indexes, # 使用全局注意力机制的 Transformer 块的索引列表
            window_size=14, # 局部注意力窗口的大小
            out_chans=prompt_embed_dim, # 输出通道数，与提示嵌入维度相同
        ),
        prompt_encoder=PromptEncoder(# 初始化提示编码器
            embed_dim=prompt_embed_dim, # 嵌入维度
            image_embedding_size=(image_embedding_size, image_embedding_size), # 图像嵌入的尺寸
            input_image_size=(image_size, image_size),# 输入图像的尺寸
            mask_in_chans=16, # 掩码输入的通道数
        ),
        mask_decoder=MaskDecoder( # 初始化掩码解码器
            num_multimask_outputs=3, # 要预测的多掩码输出的数量
            transformer=TwoWayTransformer(# 双向 Transformer 模块
                depth=2,   # Transformer 的层数
                embedding_dim=prompt_embed_dim, # 嵌入维度
                mlp_dim=2048,# MLP 层的维度
                num_heads=8,# 多头注意力机制的头数
            ),
            transformer_dim=prompt_embed_dim,  # 变压器（Transformer）的维度
            iou_head_depth=3, # IoU 预测头的深度
            iou_head_hidden_dim=256, # IoU 预测头的隐藏层维度
        ),
        pixel_mean=[123.675, 116.28, 103.53],  # 用于图像像素归一化的均值
        pixel_std=[58.395, 57.12, 57.375],# 用于图像像素归一化的标准差
    )
    sam.eval()  # 将模型设置为评估模式，关闭训练相关的操作（如 Dropout 等）
    if checkpoint is not None: # 如果提供了预训练检查点文件路径
        with open(checkpoint, "rb") as f: # 以二进制读取模式打开检查点文件
            state_dict = torch.load(f) # 加载检查点文件中的模型状态字典
        sam.load_state_dict(state_dict)# 将加载的状态字典应用到 SAM 模型中
    return sam# 返回构建好的 SAM 模型
