import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
import cv2
import math
from torch import Tensor


# 计算不确定性
def calculate_uncertainty(mask_logits):  # bchw
    """
    计算每个位置的不确定性，这里简单用 logits 绝对值衡量
    """
    return -torch.abs(mask_logits).squeeze(1)  # bhw >0


"""
def point_sample(input, point_coords, **kwargs):#input bhw/bchw  point_coords bn2

    #在输入特征图上采样指定点的特征

    # 给 input 添加通道维度
    if input.dim() == 3:#bhw
        input = input.unsqueeze(1)#b1hw
    add_dim = False# 标记是否需要添加维度
    if point_coords.dim() == 3:# 检查 point_coords 的维度是否为 3
        add_dim = True # 如果维度为 3，标记需要添加维度
        point_coords = point_coords.unsqueeze(2) # 在 point_coords 的第 2 维添加一个维度，使其维度变为 4 #bn12
    # 使用 F.grid_sample 函数进行采样
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs) # 先将 point_coords 进行变换，使其范围从 [0, 1] 变为 [-1, 1] # 因为 F.grid_sample 函数要求输入的坐标范围是 [-1, 1]
    if add_dim: # 如果之前添加了维度，现在将其移除
        output = output.squeeze(3)
    return output#bcn
"""


def extract_local_features(feature_map, point_coords, kernel_size=3):
    """
    从特征图中提取点周围的局部特征
    :param feature_map: 特征图，形状为 (B, C, H, W)
    :param point_coords: 采样点坐标，可以是张量 (B, N, 2) 或列表 [N_i, 2]，坐标为归一化值
    :return: 局部特征，形状为 (B, C, N) 或列表 [(C, N_i)]
    """
    # 给 input 添加通道维度
    if feature_map.dim() == 3:  # bhw
        feature_map = feature_map.unsqueeze(1)  # b1hw
    # print("start extract_local_features")
    if isinstance(point_coords, torch.Tensor):
        B, C, H, W = feature_map.shape
        N = point_coords.shape[1]
        # 将归一化坐标转换为特征图上的实际坐标
        x = (point_coords[..., 1] * (W - 1)).long()
        y = (point_coords[..., 0] * (H - 1)).long()
        # 计算局部区域的边界
        x_min = torch.clamp(x - kernel_size // 2, min=0)
        x_max = torch.clamp(x + kernel_size // 2 + 1, max=W)
        y_min = torch.clamp(y - kernel_size // 2, min=0)
        y_max = torch.clamp(y + kernel_size // 2 + 1, max=H)
        # 提取局部区域特征
        local_features = []
        for b in range(B):
            batch_features = []
            for n in range(N):
                local_feature = feature_map[b, :, y_min[b, n]:y_max[b, n], x_min[b, n]:x_max[b, n]].mean(dim=(1, 2))
                batch_features.append(local_feature)
            batch_features = torch.stack(batch_features, dim=1)
            local_features.append(batch_features)
        local_features = torch.stack(local_features, dim=0)
        # print("finish extract_local_features")
        return local_features
    elif isinstance(point_coords, list):
        output_list = []
        for i, current_point_coords in enumerate(point_coords):
            C, H, W = feature_map[i].shape
            N = current_point_coords.shape[0]
            x = (current_point_coords[:, 1] * (W - 1)).long()
            y = (current_point_coords[:, 0] * (H - 1)).long()
            x_min = torch.clamp(x - kernel_size // 2, min=0)
            x_max = torch.clamp(x + kernel_size // 2 + 1, max=W)
            y_min = torch.clamp(y - kernel_size // 2, min=0)
            y_max = torch.clamp(y + kernel_size // 2 + 1, max=H)
            batch_local_features = []
            for n in range(N):
                local_feature = feature_map[i, :, y_min[n]:y_max[n], x_min[n]:x_max[n]].mean(dim=(1, 2))
                batch_local_features.append(local_feature)
            batch_local_features = torch.stack(batch_local_features, dim=1)
            output_list.append(batch_local_features)
        # print("finish extract_local_features")
        return output_list


def point_sample(input, point_coords, **kwargs):  # input bhw/bchw  point_coords bn2
    """
    在输入特征图上采样指定点的特征
    """
    # 给 input 添加通道维度
    if input.dim() == 3:  # bhw
        input = input.unsqueeze(1)  # b1hw

    if isinstance(point_coords, torch.Tensor):  # 处理 point_coords 为张量 (b, n, 2) 的情况
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            current_point_coords = point_coords.unsqueeze(2)  # 在 point_coords 的第 2 维添加一个维度，使其维度变为 4 #bn12
        output = F.grid_sample(input, 2.0 * current_point_coords - 1.0, **kwargs)  # 使用 F.grid_sample 函数进行采样
        if add_dim:
            output = output.squeeze(3)
        # return output  # bcn
        return output.permute(0, 2, 1)  # bnc

    elif isinstance(point_coords, list):
        # 处理 point_coords 为列表的情况
        b = len(point_coords)
        output_list = []
        for i in range(b):
            current_point_coords = point_coords[i].unsqueeze(0).unsqueeze(2)  # 1n12
            current_input = input[i].unsqueeze(0)  # 1chw
            current_output = F.grid_sample(current_input, 2.0 * current_point_coords - 1.0, **kwargs)
            # current_output = current_output.squeeze(0).squeeze(2)  # cn
            current_output = current_output.squeeze(0).squeeze(2).permute(1, 0)  # nc
            output_list.append(current_output)
        return output_list  # list （c，<=n）


def point_sample_by_idx(input, sample_indices):
    # 给 input 添加通道维度
    if input.dim() == 3:  # bhw
        input = input.unsqueeze(1)  # b1hw

    if isinstance(sample_indices, torch.Tensor):  # 处理 point_coords 为张量 (b, n, 2) 的情况
        b, c, h, w = input.shape
        # 将 input 调整为 (b, c, h * w) 形状
        input_flat = input.view(b, c, -1)
        # 扩展 sample_indices 为 (b, c, n) 形状
        sample_indices_expanded = sample_indices.unsqueeze(1).expand(-1, c, -1)
        # 使用 gather 函数根据索引提取数据
        output = torch.gather(input_flat, 2, sample_indices_expanded)  # bcn
        output = output.transpose(1, 2)  # 将输出调整为 (b, n, c) 形状
        return output  # bnc

    elif isinstance(sample_indices, list):
        b, c = input.shape[:2]
        input_flat = input.view(b, c, -1)
        output_list = []
        for i in range(b):
            output = torch.gather(
                input_flat[i],  # c,h*w
                dim=1,
                index=sample_indices[i].unsqueeze(0).expand(c, -1)
            ).transpose(0, 1)  # <=n,c
            output_list.append(output)
        return output_list  # list <=n,c


def point_sample_img_by_idx(input, sample_indices):
    assert input.shape[0] == 1 and len(input.shape) == 4

    if isinstance(sample_indices, torch.Tensor):  # 处理 point_coords 为张量 (b, n, 2) 的情况
        b, c, h, w = input.shape
        # 将 input 调整为 (b, c, h * w) 形状
        input_flat = input.view(b, c, -1)
        # 扩展 sample_indices 为 (b, c, n) 形状
        sample_indices_expanded = sample_indices.unsqueeze(1).expand(-1, c, -1)  # bcn
        output_list = []
        B = sample_indices.shape[0]
        for i in range(B):
            current_indices = sample_indices_expanded[i]  # cn
            output = torch.gather(input_flat[0], 1, current_indices)  # cn
            output = output.transpose(0, 1)  # nc
            output_list.append(output)
        output_list = torch.stack(output_list, dim=0)  # bnc
        return output_list  # bnc

    elif isinstance(sample_indices, list):
        b, c = input.shape[:2]
        input_flat = input.view(b, c, -1)
        output_list = []
        B = len(sample_indices)
        for i in range(B):
            output = torch.gather(
                input_flat[0],  # c,h*w
                dim=1,
                index=sample_indices[i].unsqueeze(0).expand(c, -1)
            ).transpose(0, 1)  # <=n,c
            output_list.append(output)
        return output_list  # list <=n,c


def point_sample_img(input, point_coords, **kwargs):  # input bhw/bchw  point_coords bn2
    """
    在输入特征图上采样指定点的特征
    """
    # 给 input 添加通道维度
    if input.dim() == 3:  # bhw
        input = input.unsqueeze(1)  # b1hw

    if isinstance(point_coords, torch.Tensor):  # 处理 point_coords 为张量 (b, n, 2) 的情况
        add_dim = False
        if point_coords.dim() == 3:
            add_dim = True
            current_point_coords = point_coords.unsqueeze(2)  # 在 point_coords 的第 2 维添加一个维度，使其维度变为 4 #bn12
        b = point_coords.shape[0]
        output_list = []
        for i in range(b):
            current_point_coord = current_point_coords[i].unsqueeze(0)
            current_input = input  # 1chw
            current_output = F.grid_sample(current_input, 2.0 * current_point_coord - 1.0, **kwargs)
            current_output = current_output.squeeze(0)
            output_list.append(current_output)
        output = torch.stack(output_list, dim=0)
        if add_dim:
            output = output.squeeze(3)
        # return output  # bcn
        return output.permute(0, 2, 1)  # bnc

    elif isinstance(point_coords, list):
        # 处理 point_coords 为列表的情况
        b = len(point_coords)
        output_list = []
        for i in range(b):
            current_point_coords = point_coords[i].unsqueeze(0).unsqueeze(2)  # 1n12
            current_input = input  # 1chw
            current_output = F.grid_sample(current_input, 2.0 * current_point_coords - 1.0, **kwargs)
            # current_output = current_output.squeeze(0).squeeze(2)  # cn
            current_output = current_output.squeeze(0).squeeze(2).permute(1, 0)  # nc
            output_list.append(current_output)
        return output_list  # list （c，<=n）


def sampling_points(pred_mask, N, k=3, beta=0.75, training=True):
    b, _, h, w = pred_mask.shape
    # 计算不确定性
    uncertainty_map = calculate_uncertainty(pred_mask)  # bhw
    """
    if not training:
        H_step, W_step = 1 / h, 1 / w
        N = min(h * w, N)
        #uncertainty_map = calculate_uncertainty(pred_mask)#bhw
        _, idx = uncertainty_map.view(b, -1).topk(N, dim=1)#(b, N)

        points = torch.zeros(b, N, 2, dtype=torch.float, device=pred_mask.device)
        points[:, :, 0] = W_step / 2.0 + (idx % w).to(torch.float) * W_step
        points[:, :, 1] = H_step / 2.0 + (idx // w).to(torch.float) * H_step
        #visualize_sampling_points(pred_mask,points)
        return idx, points  #(b, N)  bn2
    """

    if not training:
        num_important = min(h * w, N)
        # print(f"num_important:{num_important}")
        edge_masks = edge_detection_mask((pred_mask > 0.0).float().squeeze(1))
        important_indices, important_coords, boundry_indices, boundry_coords = edge_topk_sampling_with_boundry(
            edge_masks, uncertainty_map, num_important, num_important)  # list （<=topk，2）

        # visualize_sampling_points(pred_mask,important_coords)
        return important_indices, important_coords, boundry_indices, boundry_coords  # list (<=topk)  list (<=topk,2)

    num_important = int(N * beta)
    edge_masks = edge_detection_mask((pred_mask > 0.0).float().squeeze(1))  # bhw 0-1
    important_indices, important_coords, boundry_indices, boundry_coords = edge_topk_sampling_with_boundry(edge_masks,
                                                                                                           uncertainty_map,
                                                                                                           num_important,
                                                                                                           num_important)  # list <=topk （<=topk，2）
    # boundry_indices=torch.stack(boundry_indices,dim=0)#bn
    # boundboundry_coordsry_indices=torch.stack(boundry_coords,dim=0)#bn2

    all_indices = torch.arange(h * w, device=pred_mask.device).unsqueeze(0).expand(b, -1)
    b = len(important_coords)
    combined_indices = []
    combined_coords = []
    H_step, W_step = 1 / h, 1 / w
    for i in range(b):
        num_important = important_coords[i].shape[0]
        num_random = N - num_important

        assert num_random > 0
        remaining_indices = all_indices[i].clone()
        remaining_indices[important_indices[i]] = -1
        valid_remaining_indices = remaining_indices[remaining_indices != -1]

        # 随机选择 num_random 个索引
        selected_indices = torch.randperm(valid_remaining_indices.size(0))[:num_random]
        random_indices_batch = valid_remaining_indices[selected_indices]  # num_random
        combined_indice = torch.cat([important_indices[i], random_indices_batch], dim=0)  # N
        combined_indices.append(combined_indice)

        # 获取随机采样点的坐标
        random_coords_batch = torch.zeros(num_random, 2, dtype=torch.float, device=pred_mask.device)
        random_coords_batch[:, 0] = H_step / 2.0 + (random_indices_batch // w).to(torch.float) * H_step
        random_coords_batch[:, 1] = W_step / 2.0 + (random_indices_batch % w).to(torch.float) * W_step
        combined_coord = torch.cat([important_coords[i], random_coords_batch], dim=0)  # N,2
        combined_coords.append(combined_coord)
    combined_indices = torch.stack(combined_indices, dim=0)  # bn
    combined_coords = torch.stack(combined_coords, dim=0)  # bn2

    # 合并重要性采样点和随机采样点
    # visualize_sampling_points(pred_mask,combined_coords)
    return combined_indices, combined_coords, boundry_indices, boundry_coords  # bN2


# 点头部网络
class PointHead(nn.Module):
    def __init__(self, in_channels=33, num_classes=1):
        super().__init__()
        self.fc = nn.Conv1d(in_channels, num_classes, kernel_size=1)
        # self.fc = nn.Linear(in_channels, num_classes)
        """
        nn.init.normal_(self.fc.weight, std=0.001)
        print("init PointHead fc weight")
        if self.fc.bias is not None:  # 对预测层进行权重初始化
            nn.init.constant_(self.fc.bias, 0)
            print("init PointHead fc bias")
        """

    def forward(self, fine_grained_features, coarse_features):  # b256n    b1n
        if isinstance(fine_grained_features, torch.Tensor) and isinstance(coarse_features, torch.Tensor):  # bnc
            # 处理 fine_grained_features 和 coarse_features 为张量的情况
            combined_features = torch.cat([fine_grained_features, coarse_features], dim=2).permute(0, 2, 1)  # bcn
            return self.fc(combined_features).permute(0, 2, 1)  # bn1
        elif isinstance(fine_grained_features, list) and isinstance(coarse_features, list):
            # 处理 fine_grained_features 和 coarse_features 为列表的情况
            output_list = []
            for fine, coarse in zip(fine_grained_features, coarse_features):
                fine = fine.unsqueeze(0)
                coarse = coarse.unsqueeze(0)
                combined_features = torch.cat([fine, coarse], dim=2).permute(0, 2, 1)
                output = self.fc(combined_features)
                output = output.permute(0, 2, 1).squeeze(0)
                output_list.append(output)
            return output_list  # list <=n,1
        else:
            raise ValueError("fine_grained_features 和 coarse_features 必须同时为张量或列表")
        # combined_features = torch.cat([fine_grained_features, coarse_features], dim=1)#b 257 n
        # return self.fc(combined_features)#b1n


class PointHead_try(nn.Module):
    def __init__(self, feature_dim=256):
        super(PointHead_try, self).__init__()

        # self.res_net = nn.Sequential(
        # nn.Linear(2 * feature_dim // 8 + 1, feature_dim // 8),
        # nn.GELU(),
        # nn.Linear(feature_dim // 8, 1)
        # )
        self.res_net = nn.Sequential(
            nn.Linear(2 * feature_dim // 8 + 1, feature_dim // 2),
            # nn.BatchNorm1d(feature_dim // 4),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(feature_dim // 2, feature_dim // 8),
            # nn.BatchNorm1d(feature_dim // 8),
            nn.GELU(),
            # nn.Dropout(0.2),
            nn.Linear(feature_dim // 8, 1)
        )

    def forward(self, main_features, img_samples, mask_samples):  # b256n    b1n
        if isinstance(img_samples, torch.Tensor) and isinstance(mask_samples, torch.Tensor):
            combined_feature = torch.cat([main_features, img_samples, mask_samples], dim=-1)  # [B,N,3*h_dim]
            residuals = self.res_net(combined_feature)  # BN1
            return residuals  # [B,N,1]

        elif isinstance(img_samples, list) and isinstance(mask_samples, list):
            output_list = []
            for main_feature, img_sample, mask_sample in zip(main_features, img_samples, mask_samples):
                main_feature = main_feature.unsqueeze(0)
                img_sample = img_sample.unsqueeze(0)
                mask_sample = mask_sample.unsqueeze(0)

                combined_feature = torch.cat([main_feature, img_sample, mask_sample], dim=-1)  # [1,N,3*h_dim]
                residuals = self.res_net(combined_feature)  # 1N1
                residuals = residuals.squeeze(0)  # N 1
                output_list.append(residuals)
            return output_list  # #N 1
        else:
            raise ValueError("fine_grained_features 和 coarse_features 必须同时为张量或列表")


class StandardPointHead(nn.Module):
    """
       一个点头部多层感知机，我们使用内核为 1 的一维卷积层来建模。该头部将细粒度和粗略预测特征作为输入。
       """

    def __init__(self):
        """
                从配置中解析以下属性：
                    fc_dim: 每个全连接层的输出维度
                    num_fc: 全连接层的数量
                    coarse_pred_each_layer: 如果为 True，粗略预测特征将连接到每层的输入
                """
        super(StandardPointHead, self).__init__()

        input_channels = 64  # 输入通道数
        output_channels = 1  # 输入通道数
        num_fc=3

        fc_dim_in = input_channels + output_channels  # 输入全连接层的维度
        fc_dim = 1 * fc_dim_in
        self.fc_layers = []
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)  # 创建一维卷积层作为全连接层
            # fc = nn.Linear(fc_dim_in, fc_dim)  # 创建一维卷积层作为全连接层
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim

        #self.norm1 = nn.LayerNorm(fc_dim_in)  # 创建一个层归一化层，用于对 MLP 块的输出进行归一化


        self.predictor = nn.Conv1d(fc_dim_in, output_channels, kernel_size=1, stride=1, padding=0)  # 预测层

        for layer in self.fc_layers:  # 对全连接层进行权重初始化
            weight_init.c2_msra_fill(layer)
            # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:  # 对预测层进行权重初始化
            nn.init.constant_(self.predictor.bias, 0)

    def forward(self, img_samples, img_samples_boundrys, pos_samples, pos_samples_boundrys, mask_samples,
                mask_samples_boundrys):

        if isinstance(img_samples, torch.Tensor) and isinstance(pos_samples,
                                                                torch.Tensor):  # 处理 fine_grained_features 和 coarse_features 为张量的情况
            # 处理 fine_grained_features 和 coarse_features 为列表的情况
            output_list = []
            for img_sample, img_samples_boundry, pos_sample, pos_samples_boundry, mask_sample, mask_samples_boundry in zip(
                    img_samples,
                    img_samples_boundrys,
                    pos_samples,
                    pos_samples_boundrys,
                    mask_samples,
                    mask_samples_boundrys):
                img_sample = img_sample.unsqueeze(0)  # bnc
                img_samples_boundry = img_samples_boundry.unsqueeze(0)
                pos_sample = pos_sample.unsqueeze(0)
                pos_samples_boundry = pos_samples_boundry.unsqueeze(0)
                mask_sample = mask_sample.unsqueeze(0)
                mask_samples_boundry = mask_samples_boundry.unsqueeze(0)

                x = torch.cat([img_sample, mask_sample], dim=-1)#bnc
                x=x.permute(0, 2, 1)
                for layer in self.fc_layers:
                    x = F.relu(layer(x))  # 经过 ReLU 激活函数
                #x = self.norm1(x)
                output=self.predictor(x).permute(0, 2, 1)  # bn1
                output_list.append(output.squeeze(0))


            return torch.stack(output_list, dim=0)  # list <=n，1




        elif isinstance(img_samples, list) and isinstance(pos_samples, list):
            # 处理 fine_grained_features 和 coarse_features 为列表的情况
            output_list = []
            for img_sample, img_samples_boundry, pos_sample, pos_samples_boundry, mask_sample, mask_samples_boundry in zip(
                    img_samples,
                    img_samples_boundrys,
                    pos_samples,
                    pos_samples_boundrys,
                    mask_samples,
                    mask_samples_boundrys):
                img_sample = img_sample.unsqueeze(0)
                img_samples_boundry = img_samples_boundry.unsqueeze(0)
                pos_sample = pos_sample.unsqueeze(0)
                pos_samples_boundry = pos_samples_boundry.unsqueeze(0)
                mask_sample = mask_sample.unsqueeze(0)
                mask_samples_boundry = mask_samples_boundry.unsqueeze(0)

                x = torch.cat([img_sample, mask_sample], dim=-1)  # bnc
                x = x.permute(0, 2, 1)
                for layer in self.fc_layers:
                    x = F.relu(layer(x))  # 经过 ReLU 激活函数
                #x = self.norm1(x)
                output = self.predictor(x).permute(0, 2, 1)  # bn1
                output_list.append(output.squeeze(0))
            return output_list  # list <=n，1
        else:
            raise ValueError("fine_grained_features 和 coarse_features 必须同时为张量或列表")
        # return self.predictor(x)#b1n


class LayerNorm2d(nn.Module):  # 定义一个名为 LayerNorm2d 的类，继承自 nn.Module，用于实现二维的层归一化
    def __init__(self, num_channels: int,
                 eps: float = 1e-6) -> None:  # 类的构造函数，num_channels 是输入通道数，eps 是一个小的数值，用于避免除零错误，默认为 1e-6
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))  # 定义可学习的权重参数，初始值为全 1 张量，形状为 (num_channels,)
        self.bias = nn.Parameter(torch.zeros(num_channels))  # 定义可学习的偏置参数，初始值为全 0 张量，形状为 (num_channels,)
        self.eps = eps  # 保存 eps 值

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 定义前向传播方法，输入为一个 PyTorch 张量 x
        u = x.mean(1, keepdim=True)  # 沿着通道维度计算均值，keepdim=True 表示保持维度不变，以便后续进行广播运算
        s = (x - u).pow(2).mean(1, keepdim=True)  # 计算方差，先减去均值，然后平方，再沿着通道维度求均值
        x = (x - u) / torch.sqrt(s + self.eps)  # 进行归一化操作，减去均值并除以标准差（加上 eps 避免除零）
        x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 将归一化后的结果乘以可学习的权重并加上偏置，通过广播机制实现
        return x  # 返回归一化并经过加权偏置后的结果


class PointRend(nn.Module):
    def __init__(self, head, k=3, beta=0.75, training=True):
        super().__init__()
        self.head = head
        self.k = k
        self.beta = beta
        self.training = training

        activation = nn.GELU
        transformer_dim = 256

        self.output_upscaling = nn.Sequential(  # 定义一个顺序容器，用于对输出进行上采样操作
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            # 第一个反卷积层，将通道数从 transformer_dim 减少到 transformer_dim // 4
            LayerNorm2d(transformer_dim // 4),  # 二维层归一化层，对通道维度进行归一化
            activation(),  # 应用指定的激活函数
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            # 第二个反卷积层，将通道数从 transformer_dim // 4 减少到 transformer_dim // 8
            activation(),  # 再次应用指定的激活函数
        )  # b 256 64 64 --> b 32 256 256

        """
        self.output_upscaling = nn.Sequential( # 定义一个顺序容器，用于对输出进行上采样操作
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 2, kernel_size=2, stride=2),  # 第一个反卷积层，将通道数从 transformer_dim 减少到 transformer_dim // 4
            LayerNorm2d(transformer_dim // 2), # 二维层归一化层，对通道维度进行归一化
            activation(), # 应用指定的激活函数
            nn.ConvTranspose2d(transformer_dim//2, transformer_dim // 4, kernel_size=2, stride=2),  # 第一个反卷积层，将通道数从 transformer_dim 减少到 transformer_dim // 4
            LayerNorm2d(transformer_dim // 4), # 二维层归一化层，对通道维度进行归一化
            activation(), # 应用指定的激活函数
            nn.ConvTranspose2d(transformer_dim//4, transformer_dim // 8, kernel_size=2, stride=2),  # 第一个反卷积层，将通道数从 transformer_dim 减少到 transformer_dim // 4
            LayerNorm2d(transformer_dim // 8), # 二维层归一化层，对通道维度进行归一化
            activation(), # 应用指定的激活函数
            nn.ConvTranspose2d(transformer_dim // 8, transformer_dim // 16, kernel_size=2, stride=2), # 第二个反卷积层，将通道数从 transformer_dim // 4 减少到 transformer_dim // 8
            activation(), # 再次应用指定的激活函数
        )#b 256 64 64 --> b 16 1024 1024
        """

    def forward(self, img_feature, pred_mask, output_size):
        points = sampling_points(pred_mask, 4096, self.k, self.beta, True)  # bn2

        coarse = point_sample(pred_mask, points, align_corners=False)  # bcn
        fine = point_sample(img_feature, points, align_corners=False)
        rend = self.head(fine, coarse)

        return {"rend": rend, "points": points}

    def forward_train(self, img_feature, pred_mask, valid_h=1.0, valid_w=1.0):
        points = sampling_points(pred_mask, (pred_mask.shape[-1] // 64) * (pred_mask.shape[-2] // 64), self.k,
                                 self.beta, True)  # bn2 0-1

        # coarse = point_sample(pred_mask, points, align_corners=False)#bcn
        # fine = point_sample(self.output_upscaling(img_feature).repeat(pred_mask.shape[0], 1, 1, 1), scale_points(points,valid_h,valid_w), align_corners=False)

        coarse = extract_local_features(pred_mask, points).detach()  # bcn
        fine = extract_local_features(img_feature.repeat(pred_mask.shape[0], 1, 1, 1),
                                      scale_points(points, valid_h, valid_w)).detach()
        rend = self.head(fine, coarse)  # b1n

        return {"rend": rend, "points": points}

    def forward_eval(self, img_feature, pred_mask, valid_h=1.0, valid_w=1.0):
        idx, points = sampling_points(pred_mask, (pred_mask.shape[-1] // 64) * (pred_mask.shape[-2] // 64), self.k,
                                      self.beta, False)  # list <=n（<=n,2）

        # coarse = point_sample(pred_mask, points, align_corners=False)# list c,<=n
        # fine = point_sample(self.output_upscaling(img_feature).repeat(pred_mask.shape[0], 1, 1, 1), scale_points(points,valid_h,valid_w), align_corners=False)# list c,<=n
        coarse = extract_local_features(pred_mask, points)  # bcn
        fine = extract_local_features(img_feature.repeat(pred_mask.shape[0], 1, 1, 1),
                                      scale_points(points, valid_h, valid_w))

        rend = self.head(fine, coarse)  # list 1,<=n

        return {"rend": rend, "points": points, "idx": idx}


def scale_points(points, valid_h, valid_w):
    """
    将 points 中的坐标乘以有效宽高比例

    参数:
    points (torch.Tensor or list): 形状为 (b, n, 2) 的坐标张量或列表形式，列表每个元素形状可能小于 (n, 2)
    valid_h (float): 有效高度占 1024 的比例
    valid_w (float): 有效宽度占 1024 的比例


    返回:
    torch.Tensor or list: 缩放后的坐标张量或列表形式的坐标
    """
    # 确保有效宽高比例在 [0, 1] 范围内
    valid_h = torch.clamp(torch.tensor(valid_h), 0, 1)
    valid_w = torch.clamp(torch.tensor(valid_w), 0, 1)

    if isinstance(points, torch.Tensor):
        # 处理 points 为 (b, n, 2) 张量的情况
        # 提取 x 坐标和 y 坐标
        h_coords = points[:, :, 0]
        w_coords = points[:, :, 1]

        # 乘以比例
        scaled_h = h_coords * valid_h
        scaled_w = w_coords * valid_w

        # 确保缩放后的坐标在 [0, valid_w] 和 [0, valid_h] 范围内
        scaled_h = torch.clamp(scaled_h, 0, valid_h)
        scaled_w = torch.clamp(scaled_w, 0, valid_w)

        # 重新组合坐标
        scaled_points = torch.stack([scaled_h, scaled_w], dim=-1)
        return scaled_points
    elif isinstance(points, list):
        # 处理 points 为列表形式的情况
        scaled_points_list = []
        for point in points:
            if point.numel() == 0:  # 处理空张量的情况
                scaled_points_list.append(point)
                continue
            # 提取 x 坐标和 y 坐标
            h_coords = point[:, 0]
            w_coords = point[:, 1]

            # 乘以比例
            scaled_h = h_coords * valid_h
            scaled_w = w_coords * valid_w

            # 确保缩放后的坐标在 [0, valid_w] 和 [0, valid_h] 范围内
            scaled_h = torch.clamp(scaled_h, 0, valid_h)
            scaled_w = torch.clamp(scaled_w, 0, valid_w)

            # 重新组合坐标
            scaled_point = torch.stack([scaled_h, scaled_w], dim=-1)
            scaled_points_list.append(scaled_point)
        return scaled_points_list


import time
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def visualize_sampling_points(pred_mask, combined_coords, save_folder='visualize_sampling_points'):
    """
    可视化预测掩码和采样点，并保存为图片
    :param pred_mask: 预测掩码，形状为 (b, c, h, w)
    :param combined_coords: 合并后的采样点坐标，形状为 (b, N, 2) hw
    :param save_folder: 保存图片的文件夹路径
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    b, _, h_img, w_img = pred_mask.shape
    for i in range(b):
        plt.figure(figsize=(10, 10))

        # 显示预测掩码
        mask = pred_mask[i].squeeze().cpu().numpy()
        binary_mask = (mask > 0.0).astype(np.uint8)  # 将掩码转换为二值形式
        plt.imshow(binary_mask, cmap='gray', alpha=0.5)

        # 显示采样点
        points = combined_coords[i].cpu().numpy()
        h_coords = points[:, 0] * h_img
        w_coords = points[:, 1] * w_img
        plt.scatter(w_coords, h_coords, s=10, c='r', marker='o')

        plt.title(f'Predicted Mask and Sampled Points {i + 1}')

        # 保存图片
        save_path = os.path.join(save_folder, f'sample_visualization_{time.time()}_{i + 1}.png')
        plt.savefig(save_path)
        plt.close()
        print(f'save sample_visualization_ to {save_path}')


def edge_detection_mask(pred_mask, dilation_kernel_size=3):
    """
    对输入的预测掩码进行边缘检测，并返回边缘点和边缘附近范围的点
    :param pred_mask: 预测掩码，形状为 (b, h, w) 的 torch.Tensor
    :param dilation_kernel_size: 膨胀操作的核大小，默认为 3
    :return: 边缘检测后的掩码，形状为 (b, h, w) 的 torch.Tensor
    """
    b, h, w = pred_mask.shape
    edge_masks = []

    for i in range(b):
        # 获取当前批次的掩码
        mask = pred_mask[i].cpu().numpy()
        # 将掩码转换为 uint8 类型
        mask = (mask * 255).astype(np.uint8)
        # 进行 Canny 边缘检测
        edges = cv2.Canny(mask, 50, 150)
        # 创建膨胀操作的核
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        # 对边缘进行膨胀操作
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        # 将膨胀后的边缘图转换为 0-1 掩码
        edge_mask = (dilated_edges / 255).astype(np.float32)
        # 将边缘掩码添加到列表中
        edge_masks.append(edge_mask)

    # 将列表转换为 torch.Tensor
    # edge_masks = torch.tensor(edge_masks, dtype=torch.float32, device=pred_mask.device)

    # 先将列表转换为单个 numpy.ndarray
    edge_masks_np = np.array(edge_masks)
    # 然后将 numpy.ndarray 转换为 torch.Tensor
    edge_masks = torch.tensor(edge_masks_np, dtype=torch.float32, device=pred_mask.device)

    return edge_masks  # bhw


def edge_topk_sampling(edge_masks, uncertainty_map, topk):
    """
    从边缘点中选取前 topk 大的不确定性值对应的点作为重要性采样点，并输出归一化后的坐标
    :param edge_masks: 边缘掩码，形状为 (b, h, w)，边缘点为 1，其余为 0
    :param uncertainty_map: 全部的不确定性图，形状为 (b, h, w)
    :param topk: 需要选取的前 topk 大的不确定性值的数量
    :return: 采样点的归一化坐标，形状为 (b, topk, 2)
    """
    b, h, w = edge_masks.shape
    sampling_coords = []
    original_indices = []
    for i in range(b):
        # 获取当前批次的边缘掩码和不确定性图
        edge_mask = edge_masks[i]  # hw
        uncertainty = uncertainty_map[i]  # hw

        edge_uncertainty = uncertainty[edge_mask == 1]  # 筛选出边缘点对应的不确定性值 #n
        num_edge_points = edge_uncertainty.size(0)
        # print(f"num_edge_points:{num_edge_points}")
        # 处理 topk 大于边缘点数量的情况
        effective_topk = min(topk, num_edge_points)
        # effective_topk = num_edge_points//50 #test

        # print(f"effective_topk:{effective_topk}")
        _, topk_indices = torch.topk(edge_uncertainty, k=effective_topk, dim=0)  # 获取前 topk 大的不确定性值的索引#topk

        edge_indices = torch.nonzero(edge_mask == 1)  # n，2 # 获取原始的边缘点索引
        sampling_index = edge_indices[topk_indices]  # topk，2   # 根据 topk 索引获取原始的采样点索引

        sampling_coord = sampling_index.float() / torch.tensor([h - 1, w - 1], dtype=torch.float32,
                                                               device=edge_masks.device)  # 将索引转换为归一化坐标
        sampling_coords.append(sampling_coord)

        # 计算原始的采样点索引在展平后的一维张量中的位置
        original_index = sampling_index[:, 0] * w + sampling_index[:, 1]
        original_indices.append(original_index)
    # 将采样点坐标转换为张量
    # sampling_coords = torch.stack(sampling_coords, dim=0)
    return original_indices, sampling_coords  # list b，<topk,2


def edge_topk_sampling_with_boundry(edge_masks, uncertainty_map, topk, untopk):
    """
    从边缘点中选取前 topk 大的不确定性值对应的点作为重要性采样点，并输出归一化后的坐标
    :param edge_masks: 边缘掩码，形状为 (b, h, w)，边缘点为 1，其余为 0
    :param uncertainty_map: 全部的不确定性图，形状为 (b, h, w)
    :param topk: 需要选取的前 topk 大的不确定性值的数量
    :return: 采样点的归一化坐标，形状为 (b, topk, 2)
    """
    b, h, w = edge_masks.shape
    sampling_coords = []
    original_indices = []
    sampling_coords_un = []
    original_indices_un = []
    for i in range(b):
        # 获取当前批次的边缘掩码和不确定性图
        edge_mask = edge_masks[i]  # hw
        uncertainty = uncertainty_map[i]  # hw

        edge_uncertainty = uncertainty[edge_mask == 1]  # 筛选出边缘点对应的不确定性值 #n
        num_edge_points = edge_uncertainty.size(0)

        # 处理 topk 大于边缘点数量的情况
        effective_topk = min(topk, num_edge_points)
        # effective_untopk = min(untopk,num_edge_points)
        effective_untopk = num_edge_points // 2

        _, topk_indices = torch.topk(edge_uncertainty, k=effective_topk, dim=0)  # 获取前 topk 大的不确定性值的索引#topk
        _, untopk_indices = torch.topk(-edge_uncertainty, k=effective_untopk, dim=0)  # 获取前 topk 大的不确定性值的索引#topk

        edge_indices = torch.nonzero(edge_mask == 1)  # n，2 # 获取原始的边缘点索引

        sampling_index = edge_indices[topk_indices]  # topk，2   # 根据 topk 索引获取原始的采样点索引
        sampling_index_un = edge_indices[untopk_indices]  # topk，2   # 根据 topk 索引获取原始的采样点索引

        sampling_coord = sampling_index.float() / torch.tensor([h - 1, w - 1], dtype=torch.float32,
                                                               device=edge_masks.device)  # 将索引转换为归一化坐标
        sampling_coord_un = sampling_index_un.float() / torch.tensor([h - 1, w - 1], dtype=torch.float32,
                                                                     device=edge_masks.device)  # 将索引转换为归一化坐标

        sampling_coords.append(sampling_coord)
        sampling_coords_un.append(sampling_coord_un)

        # 计算原始的采样点索引在展平后的一维张量中的位置
        original_index = sampling_index[:, 0] * w + sampling_index[:, 1]
        original_index_un = sampling_index_un[:, 0] * w + sampling_index_un[:, 1]

        original_indices.append(original_index)
        original_indices_un.append(original_index_un)
    # 将采样点坐标转换为张量
    # sampling_coords = torch.stack(sampling_coords, dim=0)
    return original_indices, sampling_coords, original_indices_un, sampling_coords_un  # list b，<topk,2


def fill_random_coords(important_coords, N, device):
    """
    填充随机采样点，使其填满每个批次所缺的数量
    :param important_coords: 重要性采样点坐标，列表形式，每个元素形状可能小于 (topk, 2)
    :param N: 每个批次需要的总采样点数
    :param device: 设备
    :return: 填充后的采样点坐标，形状为 (b, N, 2)
    """
    b = len(important_coords)
    combined_coords = []
    for i in range(b):
        num_important = important_coords[i].shape[0]
        num_random = N - num_important
        # print(f"num_important:{num_important} num_random{num_random}")
        if num_random > 0:
            random_coords = torch.rand(num_random, 2, device=device)
            combined_coord = torch.cat([important_coords[i], random_coords], dim=0)
        else:
            combined_coord = important_coords[i][:N]
        combined_coords.append(combined_coord)
    combined_coords = torch.stack(combined_coords, dim=0)
    return combined_coords


class PointRend_try(nn.Module):
    def __init__(self, head, k=3, beta=0.75, training=True, feature_dim=256, hidden_dim=128,
                 ):
        super().__init__()
        self.head = head
        self.k = k
        self.beta = beta
        self.training = training

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            # 第一个反卷积层，将通道数从 transformer_dim 减少到 transformer_dim // 4
            LayerNorm2d(256),  # 二维层归一化层，对通道维度进行归一化
            nn.GELU(),  # 应用指定的激活函数
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            # 第二个反卷积层，将通道数从 transformer_dim // 4 减少到 transformer_dim // 8
            LayerNorm2d(128),  # 二维层归一化层，对通道维度进行归一化
            nn.GELU(),  # 再次应用指定的激活函数
            nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2),
            # 第二个反卷积层，将通道数从 transformer_dim // 4 减少到 transformer_dim // 8
            LayerNorm2d(128),  # 二维层归一化层，对通道维度进行归一化
            nn.GELU(),  # 再次应用指定的激活函数
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            # 第二个反卷积层，将通道数从 transformer_dim // 4 减少到 transformer_dim // 8
            # LayerNorm2d(transformer_dim // 4), # 二维层归一化层，对通道维度进行归一化
            nn.GELU(),  # 再次应用指定的激活函数
        )

        # 主体特征提取网络
        # self.context_net = nn.Sequential(
        # nn.Linear(feature_dim, hidden_dim),
        # nn.ReLU(),
        # nn.Linear(hidden_dim, hidden_dim)
        # )

    def forward(self, img_feature, pred_mask, output_size):
        points = sampling_points(pred_mask, 4096, self.k, self.beta, True)  # bn2

        coarse = point_sample(pred_mask, points, align_corners=False)  # bcn
        fine = point_sample(img_feature, points, align_corners=False)
        rend = self.head(fine, coarse)

        return {"rend": rend, "points": points}

    def forward_train(self, image_embedding,  # [1,256,H_img,W_img]
                      mask_logits,  # [B,1,H_mask,W_mask]
                      scaled_h, scaled_w, pos_random):
        # 1.点采样
        sample_idx, sample_coords, boundry_idx, boundry_coords = sampling_points(mask_logits,
                                                                                 (mask_logits.shape[-1] // 64) * (
                                                                                         mask_logits.shape[
                                                                                             -2] // 64), self.k,
                                                                                 self.beta, True)  # bn bn2 0-1

        mask_binary = (mask_logits > 0.0).float()  # [B,1,H_mask,W_mask] 0-1
        # mask_logits_filter = mask_logits * mask_binary#[B,1,H_mask,W_mask] logit >0
        B = sample_coords.shape[0]
        N = sample_coords.shape[1]
        b_mask, c_mask, h_mask, w_mask = mask_logits.shape

        # 2.图像特征上采样
        up_features = self.upsample(image_embedding)  # [1, 32, 1024, 1024]
        ##up_features = F.interpolate(image_embedding,size=(1024,1024),mode='bilinear',align_corners=True)  # [1, 256, 1024, 1024]

        up_features = up_features[..., : scaled_h, : scaled_w].contiguous()
        if up_features.shape[-2:] != mask_logits.shape[-2:]:
            up_features = F.interpolate(up_features, size=(h_mask, w_mask), mode='bilinear',
                                        align_corners=False)  # [1, 32, H_mask, W_mask]

        # 4.采样点特征提取 !!!!!!!!!!!!!!!!!!!!!!
        # print(sample_idx[0][:10])
        # print(boundry_idx[0].shape)
        # print(boundry_idx[0][:10])
        img_samples = point_sample_img_by_idx(up_features, sample_idx)  # [B,N,32] point_sample_img_by_idx
        img_samples_boundry = point_sample_img_by_idx(up_features, boundry_idx)  # [B,N,32] point_sample_img_by_idx

        pos_samples = pos_random.compute_pe_for_sampled_points(sample_coords).detach()
        pos_samples_boundry = pos_random.compute_pe_for_sampled_points(boundry_coords)

        mask_samples = point_sample_by_idx(mask_logits, sample_idx)  # [B,N,1]
        mask_samples_boundry = point_sample_by_idx(mask_logits, boundry_idx)  # [B,N,1]

        # main_expanded = main_feature.unsqueeze(1).expand(-1, N, -1)# [B,N,32]

        rend = self.head(img_samples, img_samples_boundry, pos_samples, pos_samples_boundry, mask_samples,
                         mask_samples_boundry)  # bn1

        return {"rend": rend, "points": sample_coords, "idx": sample_idx}

    def forward_eval(self, image_embedding,  # [1,256,H_img,W_img]
                     mask_logits,  # [B,1,H_mask,W_mask]
                     scaled_h, scaled_w, pos_random):
        sample_idx, sample_coords, boundry_idx, boundry_coords = sampling_points(mask_logits, max(mask_logits.shape[-1],
                                                                                                  mask_logits.shape[
                                                                                                      -2]) // 32,
                                                                                 self.k, self.beta,
                                                                                 False)  # list <=n（<=n,2）

        mask_binary = (mask_logits > 0.0).float()  # [B,1,H_mask,W_mask] 0-1
        b_mask, c_mask, h_mask, w_mask = mask_logits.shape

        up_features = self.upsample(image_embedding)  # [1, 32, 1024, 1024]
        up_features = up_features[..., : scaled_h, : scaled_w].contiguous()
        if up_features.shape[-2:] != mask_logits.shape[-2:]:
            up_features = F.interpolate(up_features, size=(h_mask, w_mask), mode='bilinear',
                                        align_corners=False)  # [1, 32, H_mask, W_mask]

        img_samples = point_sample_img_by_idx(up_features, sample_idx)  # list <=N,h_dim
        img_samples_boundry = point_sample_img_by_idx(up_features, boundry_idx)

        pos_samples = pos_random.compute_pe_for_sampled_points(sample_coords)
        pos_samples_boundry = pos_random.compute_pe_for_sampled_points(boundry_coords)

        mask_samples = point_sample_by_idx(mask_logits, sample_idx)  # list <=N,h_dim
        mask_samples_boundry = point_sample_by_idx(mask_logits, boundry_idx)  # [B,N,1]
        # main_expanded = [main_feature[i].unsqueeze(0).expand(sample_coords[i].shape[0], -1) for i in range(len(sample_coords))]  #list <=N,32

        rend = self.head(img_samples, img_samples_boundry, pos_samples, pos_samples_boundry, mask_samples,
                         mask_samples_boundry)  # list <=n,1

        return {"rend": rend, "points": sample_coords, "idx": sample_idx}


# 定义 Transformer 层
class TransformerBoundary(nn.Module):
    def __init__(self, d_model, nhead):
        super(TransformerBoundary, self).__init__()
        self.q_proj = nn.Linear(d_model, d_model)  # 定义查询的投影层，将输入嵌入向量投影到内部维度
        self.k_proj = nn.Linear(d_model, d_model)  # 定义键的投影层，将输入嵌入向量投影到内部维度
        self.v_proj = nn.Linear(1, d_model)  # 定义值的投影层，将输入嵌入向量投影到内部维度
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        """
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.mlp=nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        """

    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        attn_output, _ = self.self_attn(q, k, v)
        return attn_output
        """
        q = q + attn_output
        q = self.norm1(q)
        mlp_output = self.mlp(q)
        q = q + mlp_output
        q = self.norm2(q)
        return q#bnc
        """


class NormalAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, num_heads):
        super(NormalAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert self.head_dim * num_heads == query_dim, "query_dim must be divisible by num_heads"

        # 用于 query 和 key 的线性投影层
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)  # 定义输出的投影层，将内部维度的向量投影回输入嵌入向量的维度

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:  # 此方法用于将输入张量按头数进行分割，实现多头注意力机制
        b, n, c = x.shape  # 获取输入张量的批次大小、序列长度和通道数
        x = x.reshape(b, n, num_heads, c // num_heads)  # 重塑输入张量，将通道维度分割为头数和每个头的通道数
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head  # 交换第 1 维和第 2 维，使得头数维度提前

    def _recombine_heads(self, x: Tensor) -> Tensor:  # 此方法用于将按头分割的张量重新组合成一个张量
        b, n_heads, n_tokens, c_per_head = x.shape  # 获取输入张量的批次大小、头数、序列长度和每个头的通道数
        x = x.transpose(1, 2)  # 交换第 1 维和第 2 维，恢复原来的维度顺序
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C  # 重塑张量，将头数和每个头的通道数合并为一个通道维度

    def forward(self, query, key, value):
        # 对 query 和 key 进行线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # 将投影后的查询、键和值按头数进行分割，实现多头注意力机制
        q = self._separate_heads(q, self.num_heads)  # B x N_heads x N_tokens x C_per_head
        k = self._separate_heads(k, self.num_heads)  # B x N_heads x N_tokens x C_per_head
        v = self._separate_heads(v, self.num_heads)

        _, _, _, c_per_head = q.shape  # 获取每个头的通道数
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens  # 计算查询和键的点积，得到注意力分数矩阵
        attn = attn / math.sqrt(c_per_head)  # 对注意力分数矩阵进行缩放，除以每个头通道数的平方根，防止点积结果过大
        attn = torch.softmax(attn, dim=-1)  # 对注意力分数矩阵在最后一个维度上进行 softmax 操作，得到注意力权重

        out = attn @ v  # 根据注意力权重对值进行加权求和，得到每个头的输出   # B x N_heads x N_tokens x 1
        out = self._recombine_heads(out)  # 将每个头的输出重新组合成一个张量 # B  x N_tokens x N_heads
        out = self.out_proj(out)

        return out  # bn1


class LogitAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads):
        super(LogitAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert self.head_dim * num_heads == query_dim, "query_dim must be divisible by num_heads"

        # 用于 query 和 key 的线性投影层
        # self.q_proj = nn.Linear(query_dim, query_dim)
        # self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(1, query_dim)
        # self.out_proj = nn.Linear(query_dim, query_dim)  # 定义输出的投影层，将内部维度的向量投影回输入嵌入向量的维度

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:  # 此方法用于将输入张量按头数进行分割，实现多头注意力机制
        b, n, c = x.shape  # 获取输入张量的批次大小、序列长度和通道数
        x = x.reshape(b, n, num_heads, c // num_heads)  # 重塑输入张量，将通道维度分割为头数和每个头的通道数
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head  # 交换第 1 维和第 2 维，使得头数维度提前

    def _recombine_heads(self, x: Tensor) -> Tensor:  # 此方法用于将按头分割的张量重新组合成一个张量
        b, n_heads, n_tokens, c_per_head = x.shape  # 获取输入张量的批次大小、头数、序列长度和每个头的通道数
        x = x.transpose(1, 2)  # 交换第 1 维和第 2 维，恢复原来的维度顺序
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C  # 重塑张量，将头数和每个头的通道数合并为一个通道维度

    def forward(self, query, key, value):
        # 对 query 和 key 进行线性投影
        q = query.clone()
        k = key.clone()
        v = self.v_proj(value)

        # 将投影后的查询、键和值按头数进行分割，实现多头注意力机制
        q = self._separate_heads(q, self.num_heads)  # B x N_heads x N_tokens x C_per_head
        k = self._separate_heads(k, self.num_heads)  # B x N_heads x N_tokens x C_per_head
        v = self._separate_heads(v, self.num_heads)  # B x N_heads x N_tokens x C_per_head

        _, _, _, c_per_head = q.shape  # 获取每个头的通道数
        attn = q @ k.permute(0, 1, 3,
                             2)  # B x N_heads x N_tokens x N_tokens  # 计算查询和键的点积，得到注意力分数矩阵 bnc * bcN=bnN *  bN1=bn1
        attn = attn / math.sqrt(c_per_head)  # 对注意力分数矩阵进行缩放，除以每个头通道数的平方根，防止点积结果过大
        attn = torch.softmax(attn, dim=-1)  # 对注意力分数矩阵在最后一个维度上进行 softmax 操作，得到注意力权重

        out = attn @ v  # 根据注意力权重对值进行加权求和，得到每个头的输出   # B x N_heads x N_tokens x 1
        out = self._recombine_heads(out)  # 将每个头的输出重新组合成一个张量 # B  x N_tokens x N_heads
        # out = self.out_proj(out)

        return out  # bn1


class FinalAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super(FinalAttention, self).__init__()
        assert query_dim == key_dim and value_dim == 1

    def forward(self, query, key, value):  # bnc bNc bN1
        _, _, c = query.shape  # 获取每个头的通道数
        attn = query @ key.permute(0, 2, 1)  # bnN
        attn = attn / math.sqrt(c)  # 对注意力分数矩阵进行缩放，除以每个头通道数的平方根，防止点积结果过大
        attn = torch.softmax(attn, dim=-1)  # 对注意力分数矩阵在最后一个维度上进行 softmax 操作，得到注意力权重
        out = attn @ value  # bn1
        return out  # bn1

