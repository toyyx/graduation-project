import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur



class DBSLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, sigma=5, smooth=1e-5):
        """
        DBD Loss：带边界距离加权的Dice Loss
        参数：
            sigma: 高斯核标准差，控制权重衰减速度（默认5）。
                   较大的 sigma 值会使权重在边界周围的衰减更慢，影响边界附近像素的权重分布。
            smooth: 平滑系数防止除零（默认1e-5）。
                    在计算 Dice 系数时，为了避免分母为零的情况，需要添加一个很小的平滑系数。
        原理：
            1. 通过形态学腐蚀提取物体边界。形态学腐蚀可以缩小物体区域，从而得到物体的边界信息。
            2. 高斯函数生成边界距离场权重图。利用高斯函数的特性，使得边界附近的像素具有更高的权重。
            3. 在 Dice Loss 计算中引入边界权重，强化边界区域误差惩罚。通过加权的方式，让模型更加关注边界区域的分割效果。
        """
        # 调用父类 nn.Module 的构造函数
        super(DBDLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        # 保存高斯核标准差
        self.sigma = sigma
        # 保存平滑系数
        self.smooth = smooth

    def focal_loss(self, pred, target):
        """
        Focal Loss实现（二分类）
        参数：
            pred: 模型输出 (N, H, W) 未经sigmoid
            target: 真实标签 (N, H, W) 值域[0,1]
        """
        # 计算概率并展平张量
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)  # (N*H*W,)
        target = target.view(-1).float()

        # 计算二元交叉熵
        bce = F.binary_cross_entropy(pred, target, reduction='none')

        # 计算Focal Loss调节因子[3]
        p_t = pred * target + (1 - pred) * (1 - target)  # 正确分类的概率
        modulating_factor = (1 - p_t) ** self.gamma

        # 应用类别权重和调节因子
        return self.alpha * modulating_factor * bce

    def _generate_weights(self, mask):
        """
        生成边界距离场权重图
        参数：
            mask: 真实标签 (N, 1, H, W) 二值化张量。
                  N 表示批量大小，1 表示通道数，H 和 W 分别表示图像的高度和宽度。
        返回：
            weight_map: 边界权重图 (N, H, W)。
                        去除了通道维度，只保留批量大小、高度和宽度信息。
        """
        # 定义边缘检测卷积核
        kernel = torch.tensor([[[
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]]], dtype=torch.float32)

        # 进行卷积操作
        boundary = F.conv2d(mask, kernel, padding=1)

        # 二值化处理得到边界掩码
        boundary_mask = (boundary > 0).float()
        """
        # 形态学腐蚀提取边界：原 mask - 腐蚀后 mask
            # 使用 3x3 的最大池化操作模拟形态学腐蚀，步长为 1，填充为 1
            eroded = F.max_pool2d(mask.float(), kernel_size=3, stride=1, padding=1)
            # 计算边界，通过原 mask 减去腐蚀后的 mask 得到，使用 clamp 函数确保结果不小于 0
            boundary = (mask.float() - eroded).clamp(min=0)  # (N, 1, H, W)
        """


        # 高斯模糊生成权重图（边界附近权重更高）
        # 计算高斯核的大小，确保覆盖 3σ 范围
        kernel_size = int(self.sigma * 4 + 1)
        # 调用 gaussian_blur 函数对边界图进行高斯模糊处理
        weight_map = gaussian_blur(
            boundary_mask,
            kernel_size=kernel_size,
            sigma=self.sigma
        )  # (N, 1, H, W)

        # 归一化到 [0, 1] 并去除通道维度
        # 找到每个样本的最大权重值，先在高度方向上取最大值，再在宽度方向上取最大值
        max_values = weight_map.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        # 为了避免除零错误，加上一个很小的数 1e-6
        weight_map = weight_map / (max_values + 1e-6)+ 1.0
        # 去除通道维度，得到 (N, H, W) 形状的权重图
        return weight_map.squeeze(1)  # (N, H, W)

    def forward(self, pred, target):
        """
        前向计算
        参数：
            pred: 模型输出 (N, 1, H, W) 未归一化。
                  模型直接输出的结果，需要进行 sigmoid 操作转换为概率值。
            target: 真实标签 (N, 1, H, W) 二值化。
                    表示每个像素的真实分类标签，是二值化的图像。
        返回：
            loss: 加权后的 DBD Loss。
                  经过加权计算得到的损失值，用于模型的训练优化。
        """
        # 生成边界权重图
        weights = self._generate_weights(target)  # (N, H, W)
        # 计算概率预测值
        # 对模型输出进行 sigmoid 操作，将其转换为概率值，并去除通道维度
        pred = torch.sigmoid(pred).squeeze(1)  # (N, H, W)????
        # 去除真实标签的通道维度，并转换为浮点型
        target = target.squeeze(1).float()  # (N, H, W)

        # 加权 Dice 系数计算
        # 计算加权交集，将预测值、真实标签和权重图对应元素相乘后求和
        intersection = (pred * target * weights).sum(dim=(1, 2))
        # 计算加权预测和，将预测值和权重图对应元素相乘后求和
        sum_pred = (pred * weights).sum(dim=(1, 2))
        # 计算加权真实和，将真实标签和权重图对应元素相乘后求和
        sum_target = (target * weights).sum(dim=(1, 2))

        # 计算 Dice 系数，使用平滑系数防止除零
        dice = (2. * intersection + self.smooth) / (sum_pred + sum_target + self.smooth)
        # 计算平均 batch 损失，用 1 减去 Dice 系数的平均值
        return 1 - dice.mean()  # 平均 batch 损失