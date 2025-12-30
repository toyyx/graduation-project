import torch
import torch.nn as nn
from torch.nn import functional as F


class EmbeddingCentricFusion(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # 通道对齐模块（每个金字塔层级独立）
        self.proj_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, 1) for _ in range(4)
        ])

    def forward(self, img_embed, pyramid_features):
        """
        :param img_embed: [B,C,H,W] 基准图像嵌入
        :param pyramid_features: 各尺度金字塔特征列表[[B,C,H/(2^i),W/(2^i)]]
        """
        B, C, H, W = img_embed.shape
        fused = img_embed.clone()
        result = []
        for i, p_feat in enumerate(pyramid_features):
            # 上采样到基准分辨率
            up_feat = F.interpolate(p_feat, (H, W), mode='bilinear')
            # 通道对齐（含可学习参数）
            aligned_feat = self.proj_convs[i](up_feat)
            # 残差相加融合[1][6]
            result.append(fused + aligned_feat)

        return result  # 输出与img_embed同维度的增强特征

class FeatureAdapter(nn.Module):
    """
    特征适配模块（包含通道对齐和空间对齐）
    输入：IA-Pyramid多尺度特征 [L1,L2,L3]
    输出：适配SAM解码器的统一特征
    """

    def __init__(self, in_channels=[256, 512, 1024], out_dim=256):
        super().__init__()
        # 通道压缩卷积层
        self.conv_l1 = nn.Conv2d(in_channels[0], out_dim, 1)
        self.conv_l2 = nn.Conv2d(in_channels[1], out_dim, 1)
        self.conv_l3 = nn.Conv2d(in_channels[2], out_dim, 1)

        # 空间对齐插值
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def forward(self, features):
        l1, l2, l3 = features

        # 通道维度对齐
        l1 = self.conv_l1(l1)  # [B,256,H/4,W/4]
        l2 = self.conv_l2(l2)  # [B,256,H/8,W/8]
        l3 = self.conv_l3(l3)  # [B,256,H/16,W/16]

        # 空间尺寸统一到最大分辨率
        l2 = self.upsample2(l2)  # H/8->H/4
        l3 = self.upsample4(l3)  # H/16->H/4

        return l1 + l2 + l3  # 多尺度特征融合


class AdaptiveDecoder(nn.Module):
    """
    改进的掩码解码器（含动态门控融合）
    输入：SAM原始特征 + IA适配特征
    """

    def __init__(self, sam_decoder, feat_dim=256):
        super().__init__()
        self.original_decoder = sam_decoder

        # 动态门控权重生成
        self.gate_conv = nn.Sequential(
            nn.Conv2d(feat_dim * 2, feat_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feat_dim // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, sam_feat, ia_feat):
        # 特征拼接生成门控权重
        concat_feat = torch.cat([sam_feat, ia_feat], dim=1)
        gate = self.gate_conv(concat_feat)  # [B,1,H,W]

        # 门控融合
        fused_feat = gate * sam_feat + (1 - gate) * ia_feat

        # 增强原始解码过程
        output_mask = self.original_decoder(fused_feat)
        return output_mask


# 使用示例
"""
假设已有：
- ia_pyramid: 交互式注意力金字塔网络
- sam_encoder: 原始SAM编码器（冻结参数）
- original_decoder: 原始SAM解码器
"""

# 初始化模块
feature_adapter = FeatureAdapter()
adaptive_decoder = AdaptiveDecoder(original_decoder)

# 前向过程
with torch.no_grad():  # 冻结编码器
    sam_feats = sam_encoder(input_img)

ia_feats = ia_pyramid(input_img)  # 获取多尺度特征
adapted_feat = feature_adapter(ia_feats)  # 特征适配

# 微调解码器
output_mask = adaptive_decoder(sam_feats, adapted_feat)

# 训练配置建议
"""
1. 优化器仅选择新添加的参数：
   optimizer = torch.optim.Adam([
       {'params': feature_adapter.parameters()},
       {'params': adaptive_decoder.parameters()}
   ], lr=1e-3)

2. 冻结原始SAM参数：
   for param in sam_encoder.parameters():
       param.requires_grad = False
   for param in original_decoder.parameters(): 
       param.requires_grad = False  # 或部分层解冻
"""