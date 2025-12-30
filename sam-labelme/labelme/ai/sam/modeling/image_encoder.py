# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type
# 从 typing 模块导入类型注解相关的工具，增强代码的可读性和可维护性
# Optional 表示该参数可以是指定类型，也可以是 None
# Tuple 用于指定元组类型
# Type 用于指定类的类型

from .common import LayerNorm2d, MLPBlock


# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):# 定义一个名为 ImageEncoderViT 的类，继承自 nn.Module，用于构建基于 Vision Transformer 的图像编码器
    def __init__(
        self,
        img_size: int = 1024, # 输入图像的尺寸，默认为 1024x1024
        patch_size: int = 16, # 图像分块的大小，即将输入图像分割成多个小块，每个小块的尺寸为 patch_size x patch_size，默认为 16
        in_chans: int = 3, # 输入图像的通道数，对于 RGB 图像，通道数为 3，默认为 3
        embed_dim: int = 768, # 分块嵌入的维度，即将每个图像块转换为一个长度为 embed_dim 的向量，默认为 768
        depth: int = 12,# Vision Transformer 的层数，即堆叠的 Transformer 块的数量，默认为 12
        num_heads: int = 12, # 每个 Transformer 块中多头注意力机制的头数，默认为 12
        mlp_ratio: float = 4.0, # MLP 隐藏层维度与嵌入维度的比例，用于确定 MLP 层的隐藏层维度，默认为 4.0
        out_chans: int = 256, # 输出特征的通道数，默认为 256
        qkv_bias: bool = True, # 如果为 True，则在查询（query）、键（key）、值（value）中添加可学习的偏置项，默认为 True
        norm_layer: Type[nn.Module] = nn.LayerNorm, # 归一化层的类型，默认为 nn.LayerNorm
        act_layer: Type[nn.Module] = nn.GELU, # 激活函数的类型，默认为 nn.GELU
        use_abs_pos: bool = True, # 如果为 True，则使用绝对位置嵌入，默认为 True
        use_rel_pos: bool = False, # 如果为 True，则在注意力图中添加相对位置嵌入，默认为 False
        rel_pos_zero_init: bool = True,  # 如果为 True，则将相对位置参数初始化为零，默认为 True
        window_size: int = 0,  # 窗口注意力块的窗口大小，默认为 0
        global_attn_indexes: Tuple[int, ...] = (),# 使用全局注意力的块的索引元组，默认为空
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super().__init__()
        self.img_size = img_size# 保存输入图像的尺寸

        self.patch_embed = PatchEmbed( # 创建一个 PatchEmbed 实例，用于将输入图像分割成多个小块并进行嵌入
            kernel_size=(patch_size, patch_size), # 卷积核的大小，与分块大小相同
            stride=(patch_size, patch_size),# 卷积的步长，与分块大小相同
            in_chans=in_chans,# 输入图像的通道数
            embed_dim=embed_dim,# 分块嵌入的维度
        )

        self.pos_embed: Optional[nn.Parameter] = None# 初始化绝对位置嵌入参数，初始值为 None
        if use_abs_pos:# 如果使用绝对位置嵌入
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter( # 用预训练图像尺寸初始化绝对位置嵌入
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim) # 创建一个可学习的参数，形状为 (1, 图像分块的行数, 图像分块的列数, 嵌入维度)，初始值全为 0
            )

        self.blocks = nn.ModuleList() # 创建一个 nn.ModuleList 用于存储多个 Transformer 块
        for i in range(depth):  # 循环 depth 次，构建 depth 个 Transformer 块
            block = Block(  # 创建一个 Block 实例，即一个 Transformer 块
                dim=embed_dim, # 输入特征的维度，即嵌入维度
                num_heads=num_heads, # 多头注意力机制的头数
                mlp_ratio=mlp_ratio,# MLP 隐藏层维度与嵌入维度的比例
                qkv_bias=qkv_bias, # 是否在查询、键、值中添加偏置项
                norm_layer=norm_layer,  # 归一化层的类型
                act_layer=act_layer, # 激活函数的类型
                use_rel_pos=use_rel_pos, # 是否使用相对位置嵌入
                rel_pos_zero_init=rel_pos_zero_init, # 是否将相对位置参数初始化为零
                window_size=window_size if i not in global_attn_indexes else 0, # 如果当前块的索引不在使用全局注意力的块的索引列表中，则使用指定的窗口大小；否则窗口大小为 0
                input_size=(img_size // patch_size, img_size // patch_size), # 输入特征的尺寸，即图像分块的行数和列数
            )
            self.blocks.append(block)# 将创建的 Transformer 块添加到 nn.ModuleList 中

        self.neck = nn.Sequential(# 创建一个 nn.Sequential 容器，用于构建颈部网络
            nn.Conv2d(# 第一个卷积层
                embed_dim,# 输入通道数，即嵌入维度
                out_chans, # 输出通道数
                kernel_size=1,  # 卷积核的大小为 1x1
                bias=False, # 不使用偏置项
            ),
            LayerNorm2d(out_chans),# 二维层归一化层，对输出特征进行归一化
            nn.Conv2d( # 第二个卷积层
                out_chans, # 输入通道数，即上一层的输出通道数
                out_chans, # 输出通道数s
                kernel_size=3, # 卷积核的大小为 3x3
                padding=1,# 填充大小为 1，保持特征图的尺寸不变
                bias=False, # 不使用偏置项
            ),
            LayerNorm2d(out_chans),# 二维层归一化层，对输出特征进行归一化
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:# 定义前向传播方法，输入为一个 PyTorch 张量 x
        x = self.patch_embed(x)# 将输入图像进行分块并嵌入，得到分块嵌入后的特征
        if self.pos_embed is not None: # 如果使用了绝对位置嵌入
            x = x + self.pos_embed # 将分块嵌入后的特征与绝对位置嵌入相加

        for blk in self.blocks:# 遍历所有的 Transformer 块
            x = blk(x) # 将特征依次通过每个 Transformer 块进行处理

        x = self.neck(x.permute(0, 3, 1, 2))
        # 调整特征的维度顺序，将通道维度移到第二个位置
        # 然后将特征通过颈部网络进行处理

        return x # 返回最终的输出特征


class Block(nn.Module): # 定义一个名为 Block 的类，继承自 nn.Module，该类表示一个 Transformer 块，支持窗口注意力机制和残差传播
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int, # 输入通道的数量，也就是输入特征的维度
        num_heads: int, # 每个 ViT 块中多头注意力机制的头数
        mlp_ratio: float = 4.0, # MLP 隐藏层维度与嵌入维度的比例，默认为 4.0
        qkv_bias: bool = True, # 如果为 True，则在查询（query）、键（key）、值（value）中添加可学习的偏置项，默认为 True
        norm_layer: Type[nn.Module] = nn.LayerNorm,# 归一化层的类型，默认为 nn.LayerNorm
        act_layer: Type[nn.Module] = nn.GELU, # 激活函数的类型，默认为 nn.GELU
        use_rel_pos: bool = False,# 如果为 True，则在注意力图中添加相对位置嵌入，默认为 False
        rel_pos_zero_init: bool = True, # 如果为 True，则将相对位置参数初始化为零，默认为 True
        window_size: int = 0, # 窗口注意力块的窗口大小。如果为 0，则使用全局注意力
        input_size: Optional[Tuple[int, int]] = None, # 输入分辨率，用于计算相对位置参数的大小。可以为 None
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)# 创建第一个归一化层，对输入特征进行归一化处理
        self.attn = Attention(# 创建一个 Attention 实例，用于执行注意力机制
            dim,# 输入特征的维度
            num_heads=num_heads,# 多头注意力机制的头数
            qkv_bias=qkv_bias,# 是否在查询、键、值中添加偏置项
            use_rel_pos=use_rel_pos, # 是否使用相对位置嵌入
            rel_pos_zero_init=rel_pos_zero_init,# 是否将相对位置参数初始化为零
            input_size=input_size if window_size == 0 else (window_size, window_size), # 如果窗口大小为 0，则使用输入分辨率；否则使用窗口大小作为输入分辨率
        )

        self.norm2 = norm_layer(dim) # 创建第二个归一化层，用于对经过注意力机制后的特征进行归一化处理
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer) # 创建一个 MLPBlock 实例，用于执行多层感知机操作
        # embedding_dim 为输入特征的维度
        # mlp_dim 为 MLP 隐藏层的维度，通过输入维度乘以 mlp_ratio 得到
        # act 为激活函数的类型

        self.window_size = window_size# 保存窗口大小

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 定义前向传播方法，输入为一个 PyTorch 张量 x
        shortcut = x # 保存输入特征，用于残差连接
        x = self.norm1(x) # 对输入特征进行第一次归一化处理
        # Window partition
        if self.window_size > 0: # 如果窗口大小大于 0，执行窗口划分操作
            H, W = x.shape[1], x.shape[2] # 获取输入特征的高度和宽度
            x, pad_hw = window_partition(x, self.window_size) # 调用 window_partition 函数将输入特征划分为多个窗口
            # pad_hw 表示填充的高度和宽度信息

        x = self.attn(x)# 将划分后的特征输入到注意力机制中进行处理
        # Reverse window partition
        if self.window_size > 0: # 如果窗口大小大于 0，执行窗口逆划分操作
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))# 调用 window_unpartition 函数将划分后的窗口特征恢复为原始的特征形状

        x = shortcut + x # 执行残差连接，将输入特征和经过注意力机制处理后的特征相加
        x = x + self.mlp(self.norm2(x))
        # 对经过注意力机制处理后的特征进行第二次归一化处理
        # 然后将其输入到 MLP 中进行处理
        # 最后再次执行残差连接

        return x


class Attention(nn.Module): # 定义一个名为 Attention 的类，继承自 nn.Module，该类实现了带有相对位置嵌入的多头注意力机制块
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int, # 输入通道的数量，也就是输入特征的维度
        num_heads: int = 8, # 注意力头的数量，默认为 8
        qkv_bias: bool = True,# 如果为 True，则在查询（query）、键（key）、值（value）中添加可学习的偏置项，默认为 True
        use_rel_pos: bool = False,# 如果为 True，则在注意力图中添加相对位置嵌入，默认为 False
        rel_pos_zero_init: bool = True,# 如果为 True，则将相对位置参数初始化为零，默认为 True
        input_size: Optional[Tuple[int, int]] = None,# 输入分辨率，用于计算相对位置参数的大小。可以为 None
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads# 保存注意力头的数量
        head_dim = dim // num_heads  # 计算每个注意力头的维度，通过输入维度除以注意力头的数量得到
        self.scale = head_dim**-0.5 # 缩放因子，用于缩放查询向量，避免点积结果过大

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # 创建一个线性层，将输入特征映射到查询（query）、键（key）、值（value）三个向量上
        # 输入维度为 dim，输出维度为 dim * 3
        self.proj = nn.Linear(dim, dim) # 创建一个线性层，用于对多头注意力机制的输出进行投影，将输出维度恢复为 dim

        self.use_rel_pos = use_rel_pos# 保存是否使用相对位置嵌入的标志
        if self.use_rel_pos: # 如果使用相对位置嵌入
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."  # 确保输入分辨率不为 None，因为需要输入分辨率来计算相对位置参数的大小
            # initialize relative positional embeddings
            # 初始化相对位置嵌入
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            # 创建一个可学习的参数，用于表示垂直方向的相对位置嵌入
            # 形状为 (2 * 输入高度 - 1, 每个头的维度)
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))
            # 创建一个可学习的参数，用于表示水平方向的相对位置嵌入
            # 形状为 (2 * 输入宽度 - 1, 每个头的维度)

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 定义前向传播方法，输入为一个 PyTorch 张量 x
        B, H, W, _ = x.shape  # 获取输入张量的批量大小 B、高度 H 和宽度 W
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # 通过线性层 self.qkv 将输入特征映射到查询、键、值三个向量上
        # 然后将结果的形状调整为 (B, H * W, 3, num_heads, C)
        # 最后交换维度，得到形状为 (3, B, num_heads, H * W, C) 的张量

        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        # 将 qkv 张量在第 0 维上拆分为查询、键、值三个张量
        # 并将形状调整为 (B * num_heads, H * W, C)

        attn = (q * self.scale) @ k.transpose(-2, -1)# 计算注意力分数
        # 首先将查询向量乘以缩放因子 self.scale
        # 然后与键向量的转置进行矩阵乘法，得到注意力分数矩阵

        if self.use_rel_pos:# 如果使用相对位置嵌入
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))# 调用 add_decomposed_rel_pos 函数，将相对位置嵌入添加到注意力分数矩阵中

        attn = attn.softmax(dim=-1) # 对注意力分数矩阵进行 softmax 操作，得到注意力权重矩阵
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        # 根据注意力权重矩阵对值向量进行加权求和
        # 然后将结果的形状调整为 (B, num_heads, H, W, C)
        # 交换维度，得到形状为 (B, H, W, num_heads, C) 的张量
        # 最后将形状调整为 (B, H, W, C)

        x = self.proj(x) # 通过线性层 self.proj 对多头注意力机制的输出进行投影

        return x  # 返回最终的输出特征


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].# 输入的张量，形状为 [批量大小, 高度, 宽度, 通道数]
        window_size (int): window size.# 窗口的大小

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C]. # 划分后的窗口张量，形状为 [批量大小 * 窗口数量, 窗口高度, 窗口宽度, 通道数]
        (Hp, Wp): padded height and width before partition  # 划分前填充后的高度和宽度
    """
    B, H, W, C = x.shape # 获取输入张量 x 的形状信息

    pad_h = (window_size - H % window_size) % window_size # 计算在高度方向上需要填充的像素数量
    pad_w = (window_size - W % window_size) % window_size # 计算在宽度方向上需要填充的像素数量
    if pad_h > 0 or pad_w > 0: # 如果在高度或宽度方向上需要填充像素
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w # 计算填充后的高度和宽度

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)# 对填充后的张量 x 进行形状调整
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # 对调整形状后的张量 x 进行维度重排
    # permute(0, 1, 3, 2, 4, 5) 将维度顺序调整为 [B, 高度方向窗口数, 宽度方向窗口数, 窗口高度, 窗口宽度, C]
    # contiguous() 确保张量在内存中是连续存储的，以便后续的 view 操作能正确执行
    # 最后将其重新调整形状为 [-1, window_size, window_size, C]，这里 -1 表示该维度的大小由其他维度自动推断
    # 最终得到的 windows 张量形状为 [B * 窗口总数, window_size, window_size, C]
    return windows, (Hp, Wp) # 返回划分后的窗口张量和填充后的高度与宽度


def window_unpartition(# 该函数的作用是将划分好的窗口重新组合成原始的序列，并去除之前填充的部分。
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].#输入的窗口张量，形状为 [批量大小 * 窗口数量, 窗口高度, 窗口宽度, 通道数]
        window_size (int): window size.#  窗口的大小
        pad_hw (Tuple): padded height and width (Hp, Wp).#填充后的高度和宽度 (Hp, Wp)
        hw (Tuple): original height and width (H, W) before padding.#填充前的原始高度和宽度 (H, W)

    Returns:
        x: unpartitioned sequences with [B, H, W, C].#重新组合且去除填充后的序列，形状为 [批量大小, 原始高度, 原始宽度, 通道数]
    """
    Hp, Wp = pad_hw # 从 pad_hw 元组中解包得到填充后的高度 Hp 和宽度 Wp
    H, W = hw # 从 hw 元组中解包得到填充前的原始高度 H 和宽度 W
    B = windows.shape[0] // (Hp * Wp // window_size // window_size) # 计算批量大小 B
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)# 对输入的窗口张量 windows 进行形状调整
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1) # 对调整形状后的张量 x 进行维度重排

    # 如果填充后的高度 Hp 大于原始高度 H 或者填充后的宽度 Wp 大于原始宽度 W
    # 说明之前进行过填充操作，需要去除填充部分
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
        # 截取填充后张量 x 的前 H 行和前 W 列，去除填充部分
        # 得到的新张量形状为 [B, H, W, 通道数]，即恢复到填充前的原始尺寸

    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:#根据查询（query）和键（key）的尺寸的相对位置，获取相对位置嵌入。
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.#查询（query）的尺寸
        k_size (int): size of key k.# 键（key）的尺寸
        rel_pos (Tensor): relative position embeddings (L, C).#相对位置嵌入张量，形状为 (L, C)，其中 L 是相对位置的数量，C 是嵌入的维度

    Returns:
        Extracted positional embeddings according to relative positions.# 根据相对位置提取的位置嵌入
    """
    # 计算最大相对距离，这是为了确定所需的相对位置嵌入的范围
    # 公式 2 * max(q_size, k_size) - 1 确保能覆盖查询和键之间所有可能的相对位置
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:# 如果相对位置嵌入的长度与最大相对距离不匹配，需要进行插值操作
        # Interpolate rel pos.
        # 首先对相对位置嵌入进行形状调整，添加一个批量维度并交换维度顺序
        # 使其符合 F.interpolate 函数的输入要求（形状为 (批量大小, 通道数, 长度)）
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, # 指定插值后的长度为最大相对距离
            mode="linear", # 使用线性插值模式
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)# 对插值后的结果进行形状调整，恢复到 (长度, 通道数) 的形状
    else:
        rel_pos_resized = rel_pos # 如果相对位置嵌入的长度已经与最大相对距离匹配，无需插值，直接使用原嵌入

    # Scale the coords with short length if shapes for q and k are different.
    # 如果查询和键的尺寸不同，需要对坐标进行缩放
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)# 生成查询的坐标，形状为 (q_size, 1)，并根据键和查询尺寸的比例进行缩放
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0) # 生成键的坐标，形状为 (1, k_size)，并根据查询和键尺寸的比例进行缩放
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0) # 计算相对坐标，通过查询坐标减去键坐标，并进行偏移以确保结果为非负
    # 根据相对坐标从调整后的相对位置嵌入中提取相应的嵌入
    return rel_pos_resized[relative_coords.long()] # 将相对坐标转换为长整型，作为索引从 rel_pos_resized 中取值


def add_decomposed_rel_pos(# 将分解的相对位置嵌入添加到注意力图中
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.#计算分解的相对位置嵌入。
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.# 注意力图，即注意力机制计算得到的分数矩阵
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).# 注意力层中的查询张量，形状为 (批量大小, 查询序列高度 * 查询序列宽度, 特征维度)
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.# 高度轴的相对位置嵌入张量，形状为 (高度方向相对位置数量, 特征维度)
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.#宽度轴的相对位置嵌入张量，形状为 (宽度方向相对位置数量, 特征维度)
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).#查询序列的空间尺寸，元组形式 (查询序列高度, 查询序列宽度)
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).#键序列的空间尺寸，元组形式 (键序列高度, 键序列宽度)

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.#添加了相对位置嵌入后的注意力图
    """
    q_h, q_w = q_size # 从 q_size 元组中解包得到查询序列的高度 q_h 和宽度 q_w
    k_h, k_w = k_size # 从 k_size 元组中解包得到键序列的高度 k_h 和宽度 k_w
    Rh = get_rel_pos(q_h, k_h, rel_pos_h) # 调用 get_rel_pos 函数，根据查询和键在高度方向的尺寸，获取高度方向的相对位置嵌入
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)# 调用 get_rel_pos 函数，根据查询和键在宽度方向的尺寸，获取宽度方向的相对位置嵌入

    B, _, dim = q.shape # 获取查询张量 q 的批量大小 B 和特征维度 dim
    r_q = q.reshape(B, q_h, q_w, dim) # 将查询张量 q 重新调整形状为 (B, q_h, q_w, dim)，以方便后续按空间维度进行操作
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)  # 使用 einsum 函数计算高度方向的相对位置得分
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)  # 使用 einsum 函数计算宽度方向的相对位置得分

    # 将注意力图 attn 重新调整形状为 (B, q_h, q_w, k_h, k_w)
    # 然后将高度方向和宽度方向的相对位置得分分别广播并加到注意力图上
    # 最后再将结果重新调整形状为 (B, q_h * q_w, k_h * k_w)
    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn # 返回添加了相对位置嵌入后的注意力图


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.#该类的作用是将输入的图像分割成多个小块（patch），并将每个小块进行嵌入操作，转换为固定维度的向量表示。
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16), # 投影层（卷积层）的卷积核大小，默认为 (16, 16)，表示将图像分割成 16x16 大小的小块
        stride: Tuple[int, int] = (16, 16), # 投影层（卷积层）的步长，默认为 (16, 16)，意味着每次卷积操作在水平和垂直方向上移动 16 个像素
        padding: Tuple[int, int] = (0, 0), # 投影层（卷积层）的填充大小，默认为 (0, 0)，即不进行填充
        in_chans: int = 3, # 输入图像的通道数，对于 RGB 图像，通道数为 3
        embed_dim: int = 768, # 每个小块嵌入后的维度，即每个小块经过卷积操作后输出的特征维度
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()
        #B*3*1024*1024----B*768*64*64
        self.proj = nn.Conv2d( # 创建一个二维卷积层作为投影层
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
            # in_chans 是输入通道数，embed_dim 是输出通道数
            # kernel_size、stride 和 padding 分别是卷积核大小、步长和填充大小
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)# 将输入的图像张量 x 通过卷积层 self.proj 进行处理
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1) # 调整输出张量的维度顺序
        return x # 返回处理后的张量
