import torch
import torch.nn as nn


class DynamicWeightLearner(nn.Module):
    """可学习权重生成模块 v1.0（基于搜索[2][7]）"""

    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        # 融合原始评分特征和iou_token_out
        self.mlp = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1对应iou_token维度
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # 输出4个权重分量
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features, iou_token):
        # 特征聚合（参考搜索[9]）
        fused = torch.cat([features.mean(dim=1), iou_token.unsqueeze(1)], dim=1)
        weights = self.softmax(self.mlp(fused))
        return weights  # [B,4]


class MaskEvaluator:
    """
    多维度掩码评选模块 v3.0
    创新特性：
    1. 多尺度特征融合评估[9]
    2. 空间关系建模（支持点/框/涂鸦/掩码）
    3. 形状复杂度分析
    4. 自适应权重调节机制
    """

    def __init__(self, image_size, prompts, scales=[0.5, 1.0, 2.0]):
        """
        :param image_size: (h, w) 输入图像尺寸
        :param prompts: dict 交互提示信息（需包含类型和坐标）
        :param scales: list 多尺度缩放因子[4]
        """
        self.h, self.w = image_size
        self.prompts = self._parse_prompts(prompts)
        self.scales = scales

        # 动态权重配置（可根据应用场景调整）
        self.weights = {
            'scale_consistency': 0.35,  # 多尺度一致性
            'spatial_relation': 0.3,  # 空间关系
            'shape_quality': 0.2,  # 形状质量
            'area_coverage': 0.15  # 区域覆盖
        }

    def _parse_prompts(self, prompts):
        """统一提示信息格式处理[7]"""
        parsed = {}
        for k, v in prompts.items():
            if k == 'points':
                parsed[k] = np.array(v).reshape(-1, 2)
            elif k == 'box':
                parsed[k] = np.array(v).reshape(2, 2)
            elif k in ['scribble', 'mask']:
                parsed[k] = np.array(v).astype(np.uint8)
        return parsed

    def evaluate(self, candidates, features):
        """
        执行多维度评估
        :param candidates: list 候选掩码列表(np.array)
        :param features: list 多尺度特征金字塔
        :return: (最佳掩码, 评分明细)
        """
        scores = []
        for mask in candidates:
            score = {
                'scale': self._calc_scale_consistency(mask, features),
                'spatial': self._calc_spatial_score(mask),
                'shape': self._calc_shape_score(mask),
                'coverage': self._calc_coverage(mask)
            }
            total = sum(score[k] * self.weights[k] for k in score)
            scores.append((total, mask, score))

        return max(scores, key=lambda x: x[0])

    def _calc_scale_consistency(self, mask, features):
        """多尺度特征一致性评估[9]"""
        similarities = []
        base_mask = cv2.resize(mask, (self.w, self.h))

        for scale, feat in zip(self.scales, features):
            scaled_mask = cv2.resize(mask, (int(self.w * scale), int(self.h * scale)))
            # 特征相似度计算（使用余弦相似度）
            feat_flat = feat.reshape(-1)
            mask_flat = scaled_mask.flatten()
            similarity = np.dot(feat_flat, mask_flat) / (np.linalg.norm(feat_flat) * np.linalg.norm(mask_flat) + 1e-8)
            similarities.append(similarity)

        return np.mean(similarities)

    def _calc_spatial_score(self, mask):
        """空间关系评分（支持多种提示类型）[4]"""
        score = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0

        # 计算掩码中心点
        M = cv2.moments(contours[0])
        cx = int(M['m10'] / (M['m00'] + 1e-8))
        cy = int(M['m01'] / (M['m00'] + 1e-8))

        # 点提示处理
        if 'points' in self.prompts:
            distances = [np.sqrt((cx - x) ** 2 + (cy - y) ** 2) for x, y in self.prompts['points']]
            score += np.exp(-min(distances) / 100)  # 距离衰减因子

        # 框提示处理
        if 'box' in self.prompts:
            x1, y1, x2, y2 = self.prompts['box'].flatten()
            box_area = (x2 - x1)\*(y2 - y1)
            intersection = max(0, min(x2, cx) - max(x1, cx)) * max(0, min(y2, cy) - max(y1, cy))
            score += intersection / (box_area + 1e-8)

        # 涂鸦/掩码提示处理
        for k in ['scribble', 'mask']:
            if k in self.prompts:
                iou = np.logical_and(mask, self.prompts[k]).sum() / np.logical_or(mask, self.prompts[k]).sum()
                score += iou

        return score / len(self.prompts)  # 归一化

    def _calc_shape_score(self, mask):
        """形状质量评估（复杂度+连续性）[7]"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0

        # 形状复杂度计算
        contour = contours[0]
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        complexity = (perimeter ** 2) / (4 * np.pi * area + 1e-8)

        # 连续性评估（凸性检测）
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / (hull_area + 1e-8)

        return 0.7 * convexity + 0.3\*(1 / complexity)

    #def _calc_coverage(self, mask):
    #    """有效区域覆盖率评估"""
    #    return np.mean(mask)

    def _calc_coverage(self, mask):
        """区域覆盖率评估（防止过大/过小区域）"""
        area_ratio = np.sum(mask) / (self.image_h * self.image_w)
        # 理想覆盖率假设为10%-50%
        return 1 - 4\*(abs(area_ratio - 0.3)) ** 2  # 抛物线型评分


class EnhancedMaskEvaluator(MaskEvaluator):
    """增强版掩码评估器 v4.0"""

    def __init__(self, image_size, prompts, feat_dim=256):
        super().__init__(image_size, prompts)
        # 可学习权重生成器（基于搜索[2][7]）
        self.weight_learner = DynamicWeightLearner(feat_dim)

        # 保留原始权重作为初始化参考
        self.base_weights = torch.tensor(list(self.weights.values()))

    def evaluate(self, candidates, features, iou_token_out):
        """
        增强评估流程：
        1. 提取多维度评分特征
        2. 动态生成权重
        3. 加权综合评分
        """
        score_features = []
        for mask in candidates:
            # 计算各维度评分（原始方法）
            scores = [
                self._calc_scale_consistency(mask, features),
                self._calc_spatial_score(mask),
                self._calc_shape_score(mask),
                self._calc_coverage(mask)
            ]
            score_features.append(torch.tensor(scores))

        # 生成动态权重（结合搜索[6][7]）
        score_tensor = torch.stack(score_features)
        dynamic_weights = self.weight_learner(
            features,  # 多尺度特征[9]
            iou_token_out  # SAM原始评分信号[8]
        )

        # 加权综合计算
        weighted_scores = (score_tensor * dynamic_weights).sum(dim=1)
        best_idx = torch.argmax(weighted_scores)

        return candidates[best_idx], {
            'weights': dynamic_weights.detach().numpy(),
            'scores': score_tensor.detach().numpy()
        }