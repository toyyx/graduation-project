# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from typing import Type

#多层感知机 块
class MLPBlock(nn.Module): # 定义一个名为 MLPBlock 的类，继承自 nn.Module，这是 PyTorch 中所有神经网络模块的基类
    def __init__(
        self,
        embedding_dim: int,# 嵌入维度，即输入特征的维度
        mlp_dim: int,# MLP（多层感知机）隐藏层的维度
        act: Type[nn.Module] = nn.GELU,# 激活函数的类型，默认为 GELU（高斯误差线性单元）激活函数
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim) # 定义第一个线性层，将输入的 embedding_dim 维度特征映射到 mlp_dim 维度
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)# 定义第二个线性层，将 mlp_dim 维度的特征映射回 embedding_dim 维度
        self.act = act() # 实例化激活函数

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 定义前向传播方法，输入为一个 PyTorch 张量 x
        return self.lin2(self.act(self.lin1(x))) # 前向传播过程：先通过 lin1 线性层，再经过激活函数，最后通过 lin2 线性层得到输出

#二维层归一化
# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module): # 定义一个名为 LayerNorm2d 的类，继承自 nn.Module，用于实现二维的层归一化
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None: # 类的构造函数，num_channels 是输入通道数，eps 是一个小的数值，用于避免除零错误，默认为 1e-6
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels)) # 定义可学习的权重参数，初始值为全 1 张量，形状为 (num_channels,)
        self.bias = nn.Parameter(torch.zeros(num_channels)) # 定义可学习的偏置参数，初始值为全 0 张量，形状为 (num_channels,)
        self.eps = eps # 保存 eps 值

    def forward(self, x: torch.Tensor) -> torch.Tensor: # 定义前向传播方法，输入为一个 PyTorch 张量 x
        u = x.mean(1, keepdim=True)# 沿着通道维度计算均值，keepdim=True 表示保持维度不变，以便后续进行广播运算
        s = (x - u).pow(2).mean(1, keepdim=True) # 计算方差，先减去均值，然后平方，再沿着通道维度求均值
        x = (x - u) / torch.sqrt(s + self.eps)# 进行归一化操作，减去均值并除以标准差（加上 eps 避免除零）
        x = self.weight[:, None, None] * x + self.bias[:, None, None] # 将归一化后的结果乘以可学习的权重并加上偏置，通过广播机制实现
        return x # 返回归一化并经过加权偏置后的结果
