# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore

from typing import Any, Dict, List, Optional, Tuple

from .modeling import Sam
from .predictor import SamPredictor
from .utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)


class SamAutomaticMaskGenerator:
    def __init__(
        self,
        model: Sam,# 用于掩码预测的 SAM 模型
        points_per_side: Optional[int] = 32,# 沿图像一侧采样的点数。总点数为 points_per_side 的平方。
            # 如果为 None，则必须通过 point_grids 显式提供点采样
        points_per_batch: int = 64,# 模型同时处理的点数。数值越高可能速度越快，但会使用更多的 GPU 内存
        pred_iou_thresh: float = 0.88,# 基于模型预测的掩码质量的过滤阈值，范围在 [0, 1] 之间
        stability_score_thresh: float = 0.95, # 基于掩码稳定性得分的过滤阈值，范围在 [0, 1] 之间。
        # 稳定性得分衡量的是模型掩码预测在二值化阈值变化时的稳定性
        stability_score_offset: float = 1.0,# 计算稳定性得分时用于偏移二值化阈值的量
        box_nms_thresh: float = 0.7, # 非极大值抑制（NMS）用于过滤重复掩码的边界框 IoU 阈值
        crop_n_layers: int = 0, # 如果大于 0，将在图像的裁剪区域上再次运行掩码预测。
        # 该值设置要运行的层数，每层的图像裁剪数量为 2**i_layer
        crop_nms_thresh: float = 0.7, # 不同裁剪区域之间用于过滤重复掩码的非极大值抑制的边界框 IoU 阈值
        crop_overlap_ratio: float = 512 / 1500, # 设置裁剪区域的重叠程度。在第一层裁剪中，裁剪区域将按此比例重叠
        crop_n_points_downscale_factor: int = 1, # 第 n 层采样的每侧点数将按 crop_n_points_downscale_factor**n 进行缩放
        point_grids: Optional[List[np.ndarray]] = None, # 一个包含显式点采样网格的列表，坐标归一化到 [0, 1]。
        # 列表中的第 n 个网格用于第 n 层裁剪。与 points_per_side 互斥
        min_mask_region_area: int = 0, # 如果大于 0，将进行后处理以移除面积小于该值的掩码中的不连通区域和孔洞。需要 opencv
        output_mode: str = "binary_mask", # 掩码的返回形式。可以是 'binary_mask'、'uncompressed_rle' 或 'coco_rle'。
        # 'coco_rle' 需要 pycocotools。对于高分辨率图像，'binary_mask' 可能会占用大量内存
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.
        #使用 SAM 模型为整个图像生成掩码。
        #在图像上生成一个点提示网格，然后过滤掉低质量和重复的掩码。
        #默认设置是为具有 ViT - H 骨干网络的 SAM 模型选择的。

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff when
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != ( # 确保 points_per_side 和 point_grids 中恰好有一个被提供
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None: # 如果提供了 points_per_side，则构建所有层的点网格
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None: # 如果提供了 point_grids，则直接使用
            self.point_grids = point_grids
        else: # 两者都未提供时抛出错误
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [ # 确保 output_mode 是合法的
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401 # 如果输出模式为 'coco_rle'，导入 pycocotools 中的 mask 模块

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401  # 如果需要移除小区域，导入 opencv

        self.predictor = SamPredictor(model)  # 创建 SamPredictor 实例用于预测
        self.points_per_batch = points_per_batch # 保存同时处理的点数
        self.pred_iou_thresh = pred_iou_thresh    # 保存预测 IoU 阈值
        self.stability_score_thresh = stability_score_thresh # 保存稳定性得分阈值
        self.stability_score_offset = stability_score_offset# 保存稳定性得分偏移量
        self.box_nms_thresh = box_nms_thresh# 保存边界框 NMS 阈值
        self.crop_n_layers = crop_n_layers  # 保存裁剪层数
        self.crop_nms_thresh = crop_nms_thresh # 保存裁剪区域间的 NMS 阈值
        self.crop_overlap_ratio = crop_overlap_ratio# 保存裁剪区域重叠比例
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor # 保存每侧点数的缩放因子
        self.min_mask_region_area = min_mask_region_area# 保存最小掩码区域面积
        self.output_mode = output_mode# 保存输出模式

    @torch.no_grad()
    def generate(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.
        #为给定的图像生成掩码。

        Arguments:
          image (np.ndarray): The image to generate masks for, in HWC uint8 format.
           #要生成掩码的图像，格式为 HWC 的 uint8 数组

        Returns:
           list(dict(str, any)): A list over records for masks. Each record is
             a dict containing the following keys:
               segmentation (dict(str, any) or np.ndarray): The mask. If
                 output_mode='binary_mask', is an array of shape HW. Otherwise,
                 is a dictionary containing the RLE.
                #掩码。如果 output_mode 为 'binary_mask'，则是一个形状为 HW 的数组；
               否则是一个包含游程编码（RLE）的字典
               bbox (list(float)): The box around the mask, in XYWH format.
               #掩码的边界框，格式为 XYWH
               area (int): The area in pixels of the mask.
                #掩码的像素面积
               predicted_iou (float): The model's own prediction of the mask's
                 quality. This is filtered by the pred_iou_thresh parameter.
                #模型对掩码质量的预测值，会根据 pred_iou_thresh 参数进行过滤
               point_coords (list(list(float))): The point coordinates input
                 to the model to generate this mask.
                #输入到模型以生成此掩码的点坐标
               stability_score (float): A measure of the mask's quality. This
                 is filtered on using the stability_score_thresh parameter.
                 #掩码质量的衡量指标，会根据 stability_score_thresh 参数进行过滤
               crop_box (list(float)): The crop of the image used to generate
                 the mask, given in XYWH format.
                #用于生成此掩码的图像裁剪区域，格式为 XYWH
        """

        # Generate masks
        mask_data = self._generate_masks(image)# 生成掩码数据

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0: # 如果需要移除小区域，进行后处理
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        # 根据输出模式对掩码进行编码
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]  # 如果输出模式为 'coco_rle'，将 RLE 编码转换为 COCO 格式的 RLE 编码
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]] # 如果输出模式为 'binary_mask'，将 RLE 编码转换为二进制掩码数组
        else:
            mask_data["segmentations"] = mask_data["rles"] # 否则直接使用 RLE 编码

        # Write mask records
        curr_anns = [] # 构建掩码记录列表
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks(self, image: np.ndarray) -> MaskData:
        """
            为输入的图像生成掩码数据。首先将图像进行裁剪，在每个裁剪区域上生成掩码，
            最后合并并去除不同裁剪区域间重复的掩码。

            参数:
            image (np.ndarray): 输入的图像，格式为 HWC 的 numpy 数组。

            返回:
            MaskData: 包含生成的掩码相关数据，如掩码、边界框、预测 IoU 等。
        """
        orig_size = image.shape[:2] # 获取原始图像的高度和宽度
        crop_boxes, layer_idxs = generate_crop_boxes(  # 生成裁剪框和对应的裁剪层索引
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()# 初始化一个 MaskData 对象用于存储所有裁剪区域生成的掩码数据
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)  # 处理当前裁剪区域，生成该区域的掩码数据
            data.cat(crop_data) # 将当前裁剪区域的掩码数据合并到总的数据中

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1: # 如果有多个裁剪区域，去除不同裁剪区域之间重复的掩码
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"]) # 优先保留来自较小裁剪区域的掩码，这里用裁剪框面积的倒数作为得分
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(# 使用非极大值抑制（NMS）去除重复的掩码
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories # 类别信息，这里统一设为 0
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)  # 过滤掉被 NMS 去除的掩码数据

        data.to_numpy() # 将数据转换为 numpy 数组
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        """
           处理图像的单个裁剪区域，生成该区域的掩码数据。

           参数:
           image (np.ndarray): 原始图像，格式为 HWC 的 numpy 数组。
           crop_box (List[int]): 裁剪框，格式为 [x0, y0, x1, y1]。
           crop_layer_idx (int): 当前裁剪区域所属的层索引。
           orig_size (Tuple[int, ...]): 原始图像的尺寸，格式为 (height, width)。

           返回:
           MaskData: 该裁剪区域生成的掩码相关数据。
        """
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box # 从裁剪框中获取坐标信息
        cropped_im = image[y0:y1, x0:x1, :]# 裁剪图像
        cropped_im_size = cropped_im.shape[:2] # 获取裁剪后图像的尺寸
        self.predictor.set_image(cropped_im) # 设置预测器的图像为裁剪后的图像

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1] # 计算用于该裁剪区域的采样点的缩放比例
        points_for_image = self.point_grids[crop_layer_idx] * points_scale# 根据当前裁剪层的点网格和缩放比例，得到该裁剪区域的采样点

        # Generate masks for this crop in batches
        data = MaskData() # 初始化一个 MaskData 对象用于存储该裁剪区域生成的掩码数据
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):# 按批次处理采样点
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size) # 处理当前批次的采样点，生成该批次的掩码数据
            data.cat(batch_data) # 将当前批次的掩码数据合并到该裁剪区域的总数据中
            del batch_data
        self.predictor.reset_image() # 重置预测器的图像

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(# 去除该裁剪区域内重复的掩码
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)  # 过滤掉被 NMS 去除的掩码数据

        # Return to the original image frame
        # 将裁剪区域内的边界框和点坐标恢复到原始图像的坐标系
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))]) # 为每个掩码设置对应的裁剪框

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        """
            处理一批采样点，生成该批次的掩码数据，并进行过滤和后处理。

            参数:
            points (np.ndarray): 一批采样点，格式为 (N, 2)，N 为点的数量。
            im_size (Tuple[int, ...]): 当前处理图像的尺寸，格式为 (height, width)。
            crop_box (List[int]): 裁剪框，格式为 [x0, y0, x1, y1]。
            orig_size (Tuple[int, ...]): 原始图像的尺寸，格式为 (height, width)。

            返回:
            MaskData: 该批次生成的掩码相关数据。
        """
        orig_h, orig_w = orig_size# 获取原始图像的高度和宽度

        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size) # 将采样点转换为预测器所需的坐标格式
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device) # 将转换后的点坐标转换为 torch 张量，并放到预测器所在的设备上
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device) # 为每个点设置标签为 1，表示前景点
        masks, iou_preds, _ = self.predictor.predict_torch( # 使用预测器预测掩码和 IoU 分数
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=True,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData( # 使用预测器预测掩码和 IoU 分数
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks

        # Filter by predicted IoU
        if self.pred_iou_thresh > 0.0:  # 根据预测的 IoU 分数过滤掩码
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        # 计算掩码的稳定性分数
        data["stability_score"] = calculate_stability_score(
            data["masks"], self.predictor.model.mask_threshold, self.stability_score_offset
        )
        if self.stability_score_thresh > 0.0:  # 根据稳定性分数过滤掩码
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        # 对掩码进行阈值处理，得到二值掩码
        data["masks"] = data["masks"] > self.predictor.model.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"]) # 根据二值掩码计算边界框

        # Filter boxes that touch crop boundaries
        # 过滤掉接触裁剪边界的边界框对应的掩码
        keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)  # 将裁剪区域内的掩码恢复到原始图像的尺寸
        data["rles"] = mask_to_rle_pytorch(data["masks"]) # 将掩码转换为游程编码（RLE）格式
        del data["masks"]

        return data

    @staticmethod
    def postprocess_small_regions(
        mask_data: MaskData, min_area: int, nms_thresh: float
    ) -> MaskData:
        """
           去除掩码中的小的不连通区域和孔洞，然后重新运行边界框的非极大值抑制（NMS）
           以去除任何新产生的重复掩码。

           参数:
           mask_data (MaskData): 包含掩码数据的对象。
           min_area (int): 最小区域面积阈值，小于该面积的区域将被去除。
           nms_thresh (float): 非极大值抑制的 IoU 阈值。

           返回:
           MaskData: 处理后的掩码数据。
        """
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates.

        Edits mask_data in place.

        Requires open-cv as a dependency.
        """
        if len(mask_data["rles"]) == 0:# 如果没有掩码数据，直接返回
            return mask_data

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for rle in mask_data["rles"]:# 遍历每个掩码的 RLE 编码
            mask = rle_to_mask(rle) # 将 RLE 编码转换为二值掩码

            mask, changed = remove_small_regions(mask, min_area, mode="holes")    # 去除掩码中的小的孔洞
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")# 去除掩码中的小的不连通区域
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0)) # 将处理后的掩码转换为 torch 张量并添加到新掩码列表中
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged)) # 为未改变的掩码设置分数为 1，改变的掩码设置分数为 0

        # Recalculate boxes and remove any new duplicates
        masks = torch.cat(new_masks, dim=0)  # 将所有新掩码拼接成一个张量
        boxes = batched_mask_to_box(masks)  # 根据新掩码计算边界框
        keep_by_nms = batched_nms( # 使用非极大值抑制去除重复的边界框
            boxes.float(),
            torch.as_tensor(scores),
            torch.zeros_like(boxes[:, 0]),  # categories
            iou_threshold=nms_thresh,
        )

        # Only recalculate RLEs for masks that have changed
        for i_mask in keep_by_nms: # 仅对发生改变的掩码重新计算 RLE 编码
            if scores[i_mask] == 0.0:
                mask_torch = masks[i_mask].unsqueeze(0)
                mask_data["rles"][i_mask] = mask_to_rle_pytorch(mask_torch)[0]
                mask_data["boxes"][i_mask] = boxes[i_mask]  # update res directly # 直接更新边界框
        mask_data.filter(keep_by_nms) # 过滤掉被 NMS 去除的掩码数据

        return mask_data
