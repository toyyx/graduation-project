# ------------------------------------------------------------------------
# Grounding DINO
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _get_clones(module, N, layer_share=False):
    # import ipdb; ipdb.set_trace()
    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack((sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3).flatten(2)
        return sin_x

    pos_res = [sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res


def gen_encoder_output_proposals(
    memory: Tensor, memory_padding_mask: Tensor, spatial_shapes: Tensor, learnedwh=None
):
    """
    Input:
        - memory: bs, \sum{hw}, d_model   编码器的输出特征，形状为 (批次大小, 所有特征层的像素总数, 特征维度)
        - memory_padding_mask: bs, \sum{hw}  编码器输出特征的填充掩码，形状为 (批次大小, 所有特征层的像素总数)，用于标记填充位置
        - spatial_shapes: nlevel, 2 每个特征层的空间形状，形状为 (特征层数量, 2)，第二维的两个值分别表示高度和宽度
        - learnedwh: 2 可学习的宽高参数，形状为 (2,)
    Output:
        - output_memory: bs, \sum{hw}, d_model 处理后的编码器输出特征，形状为 (批次大小, 所有特征层的像素总数, 特征维度)
        - output_proposals: bs, \sum{hw}, 4 生成的边界框提议，形状为 (批次大小, 所有特征层的像素总数, 4)，4 表示边界框的坐标信息
    """
    N_, S_, C_ = memory.shape   # 获取编码器输出特征的形状，N_ 为批次大小，S_ 为所有特征层的像素总数，C_ 为特征维度
    proposals = []  # 用于存储每个特征层生成的边界框提议
    _cur = 0 # 记录当前处理到的特征层的起始索引
    for lvl, (H_, W_) in enumerate(spatial_shapes):   # 遍历每个特征层
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(N_, H_, W_, 1)  # 从填充掩码中提取当前特征层的掩码，并调整形状为 (N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1) # 计算每个样本在当前特征层的有效高度
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)  # 计算每个样本在当前特征层的有效宽度

        # import ipdb; ipdb.set_trace()

        grid_y, grid_x = torch.meshgrid( # 生成网格坐标，grid_y 是高度方向的坐标，grid_x 是宽度方向的坐标
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # H_, W_, 2   # 将宽度和高度方向的坐标拼接在一起，形状为 (H_, W_, 2)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2) # 计算缩放比例，将有效宽度和有效高度拼接在一起，并调整形状为 (N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale  # 对网格坐标进行归一化处理，将其缩放到 [0, 1] 区间

        if learnedwh is not None: # 如果提供了可学习的宽高参数，使用该参数生成边界框的宽高
            # import ipdb; ipdb.set_trace()
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl) # 如果没有提供可学习的宽高参数，使用固定的宽高值 0.05 乘以 2 的 lvl 次方

        # scale = torch.cat([W_[None].unsqueeze(-1), H_[None].unsqueeze(-1)], 1).view(1, 1, 1, 2).repeat(N_, 1, 1, 1)
        # grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        # wh = torch.ones_like(grid) / scale
        # 将归一化后的网格坐标和宽高信息拼接在一起，得到边界框提议
        # 并调整形状为 (N_, -1, 4)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)# 将当前特征层的边界框提议添加到列表中
        _cur += H_ * W_ # 更新当前处理到的特征层的起始索引
    # import ipdb; ipdb.set_trace()
    output_proposals = torch.cat(proposals, 1) # 将所有特征层的边界框提议拼接在一起，形状为 (N_, \sum{hw}, 4)
    output_fix = torch.zeros_like(output_proposals, dtype=output_proposals.dtype, device=output_proposals.device)
    #shape = output_proposals.shape[:-1] + (1,) # 获取 output_proposals 除最后一维的形状
    #output_ratio = torch.ones(shape, dtype=output_proposals.dtype, device=output_proposals.device)

    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(# 生成一个掩码，标记边界框提议中所有坐标值都在 (0.01, 0.99) 区间内的位置
        -1, keepdim=True
    )
    output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid   # 对边界框提议应用逆 sigmoid 函数
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))   # 将填充位置的边界框提议值设为正无穷
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))   # 将无效的边界框提议值设为正无穷

    epsilon = 1e-8
    output_fix = torch.log((output_fix + epsilon)/ (1 - output_fix + epsilon))  # unsigmoid   # 对边界框提议应用逆 sigmoid 函数
    output_fix = output_fix.masked_fill(memory_padding_mask.unsqueeze(-1),float("inf"))  # 将填充位置的边界框提议值设为正无穷
    output_fix = output_fix.masked_fill(~output_proposals_valid, float("inf"))  # 将无效的边界框提议值设为正无穷

    #output_ratio = torch.log((output_ratio + epsilon) / (1 - output_ratio + epsilon))  # unsigmoid   # 对边界框提议应用逆 sigmoid 函数
    #output_ratio = output_ratio.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))  # 将填充位置的边界框提议值设为正无穷
    #output_ratio = output_ratio.masked_fill(~output_proposals_valid, float("inf"))  # 将无效的边界框提议值设为正无穷
    ####

    output_memory = memory # 复制编码器输出特征
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))   # 将填充位置的编码器输出特征值设为 0
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0)) # 将无效的编码器输出特征值设为 0

    # output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    # output_memory = output_memory.masked_fill(~output_proposals_valid, float('inf'))

    #return output_memory, output_proposals, output_fix, output_ratio
    return output_memory, output_proposals, output_fix


class RandomBoxPerturber:
    def __init__(
        self, x_noise_scale=0.2, y_noise_scale=0.2, w_noise_scale=0.2, h_noise_scale=0.2
    ) -> None:
        self.noise_scale = torch.Tensor(
            [x_noise_scale, y_noise_scale, w_noise_scale, h_noise_scale]
        )

    def __call__(self, refanchors: Tensor) -> Tensor:
        nq, bs, query_dim = refanchors.shape
        device = refanchors.device

        noise_raw = torch.rand_like(refanchors)
        noise_scale = self.noise_scale.to(device)[:query_dim]

        new_refanchors = refanchors * (1 + (noise_raw - 0.5) * noise_scale)
        return new_refanchors.clamp_(0, 1)


def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, no_reduction=False
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if no_reduction:
        return loss

    return loss.mean(1).sum() / num_boxes


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_activation_fn(activation, d_model=256, batch_dim=0):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode='floor')) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class ContrastiveEmbed(nn.Module):
    def __init__(self, max_text_len=256):
        """
        Args:
            max_text_len: max length of text.
        """
        super().__init__()
        self.max_text_len = max_text_len

    def forward(self, x, text_dict):
        """_summary_

        Args:
            x (_type_): _description_
            text_dict (_type_): _description_
            {
                'encoded_text': encoded_text, # bs, 195, d_model
                'text_token_mask': text_token_mask, # bs, 195
                        # True for used tokens. False for padding tokens
            }
        Returns:
            _type_: _description_
        """
        assert isinstance(text_dict, dict)

        y = text_dict["encoded_text"]
        text_token_mask = text_dict["text_token_mask"]

        res = x @ y.transpose(-1, -2)
        res.masked_fill_(~text_token_mask[:, None, :], float("-inf"))

        # padding to max_text_len
        new_res = torch.full((*res.shape[:-1], self.max_text_len), float("-inf"), device=res.device)
        new_res[..., : res.shape[-1]] = res

        return new_res
