# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from segment_anything.modeling import Sam

from typing import Optional, Tuple

from .utils.transforms import ResizeLongestSide


class SamPredictor:
    def __init__(
        self,
        sam_model: Sam, # 用于掩码预测的 SAM 模型实例
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.
        # 使用 SAM 模型为图像计算图像嵌入，然后在给定提示的情况下允许重复、高效地进行掩码预测。

        Arguments:
          sam_model (Sam): The model to use for mask prediction. # 用于掩码预测的 SAM 模型
        """
        super().__init__()
        self.model = sam_model# 保存传入的 SAM 模型
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)  # 创建一个 ResizeLongestSide 实例，用于将图像的最长边调整为模型期望的尺寸
        self.reset_image() # 重置图像相关信息，将图像状态设置为未设置

    def set_image(
        self,
        image: np.ndarray, # 用于计算掩码的图像，期望是 HWC（高度、宽度、通道）格式的 uint8 类型数组，像素值范围在 [0, 255]
        image_format: str = "RGB", # 图像的颜色格式，取值为 ['RGB', 'BGR']
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.#为提供的图像计算图像嵌入，使得可以使用 'predict' 方法进行掩码预测。

        Arguments:
          image (np.ndarray): The image for calculating masks. Expects an
            image in HWC uint8 format, with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        assert image_format in [ # 检查图像格式是否为 'RGB' 或 'BGR'，如果不是则抛出异常
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format: # 如果图像格式与模型期望的格式不一致，则反转通道顺序
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        # 将图像转换为模型期望的格式
        input_image = self.transform.apply_image(image)# 应用 ResizeLongestSide 变换，将图像的最长边调整为模型期望的尺寸
        input_image_torch = torch.as_tensor(input_image, device=self.device) # 将 NumPy 数组转换为 PyTorch 张量，并将其放置在与模型相同的设备上
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :] # 调整张量的维度顺序为 BCHW（批次、通道、高度、宽度），并确保内存连续

        self.set_torch_image(input_image_torch, image.shape[:2]) # 调用 set_torch_image 方法，传入转换后的 PyTorch 张量和原始图像的尺寸

    @torch.no_grad()
    def set_torch_image(
        self,
        transformed_image: torch.Tensor, # 输入图像，形状为 1x3xHxW，已经通过 ResizeLongestSide 进行了变换
        original_image_size: Tuple[int, ...],# 图像在变换之前的尺寸，格式为 (H, W)
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method. Expects the input
        image to be already transformed to the format expected by the model.
        # 为提供的图像计算图像嵌入，使得可以使用 'predict' 方法进行掩码预测。期望输入图像已经转换为模型期望的格式。

        Arguments:
          transformed_image (torch.Tensor): The input image, with shape
            1x3xHxW, which has been transformed with ResizeLongestSide.
            #输入图像，形状为 1x3xHxW，已经通过 ResizeLongestSide 进行了变换
          original_image_size (tuple(int, int)): The size of the image
            before transformation, in (H, W) format.
            #图像在变换之前的尺寸，格式为 (H, W)
        """
        assert (  # 检查输入图像的形状是否符合要求，形状应为 1x3xHxW，且最长边应等于模型期望的尺寸
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        self.reset_image() # 重置图像相关信息，将图像状态设置为未设置

        self.original_size = original_image_size# 保存原始图像的尺寸
        self.input_size = tuple(transformed_image.shape[-2:])# 保存输入图像（变换后）的尺寸
        input_image = self.model.preprocess(transformed_image)# 对输入图像进行预处理，如归一化、填充等操作
        self.features = self.model.image_encoder(input_image) # 将预处理后的图像输入到模型的图像编码器中，计算图像嵌入
        self.is_image_set = True# 将图像设置状态标记为已设置

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None, # 输入到模型的点提示数组，形状为 Nx2，其中每个点以像素为单位表示为 (X, Y) 坐标，可为 None
        point_labels: Optional[np.ndarray] = None, # 点提示的标签数组，长度为 N，1 表示前景点，0 表示背景点，可为 None
        box: Optional[np.ndarray] = None,  # 输入到模型的框提示数组，长度为 4，格式为 XYXY（左上角和右下角坐标），可为 None
        mask_input: Optional[np.ndarray] = None, # 输入到模型的低分辨率掩码，通常来自上一次预测迭代，形状为 1xHxW，对于 SAM 模型，H = W = 256，可为 None
        multimask_output: bool = True, # 是否让模型返回三个掩码。对于模糊的输入提示（如单个点击），返回多个掩码通常能得到更好的结果；
        # 对于明确的提示（如多个输入提示），设置为 False 可能得到更好的结果
        return_logits: bool = False, # 是否返回未经过阈值处理的掩码对数概率，而不是二进制掩码
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.
        #使用当前设置的图像，为给定的输入提示预测掩码。

        Arguments:
          point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (np.ndarray or None): A length N array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
            #点提示的标签数组，长度为 N，1 表示前景点，0 表示背景点
          box (np.ndarray or None): A length 4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form 1xHxW, where
            for SAM, H=W=256.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
             #输出的掩码数组，格式为 CxHxW，其中 C 是掩码的数量，(H, W) 是原始图像的尺寸
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
            #长度为 C 的数组，包含模型对每个掩码质量的预测
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
            # 形状为 CxHxW 的数组，其中 C 是掩码的数量，H = W = 256。这些低分辨率的对数概率可以作为掩码输入传递给后续迭代
        """
        if not self.is_image_set: # 检查是否已经设置了图像，如果未设置则抛出运行时错误
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Transform input prompts
        # 初始化转换后的输入提示为 None
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        if point_coords is not None:# 如果提供了点提示
            assert (# 确保同时提供了点提示的标签，否则抛出异常
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size) # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device) # 将变换后的点坐标转换为 PyTorch 张量，并放置在与模型相同的设备上
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device) # 将点标签转换为 PyTorch 张量，并放置在与模型相同的设备上
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :] # 在第 0 维添加一个维度，以符合批量输入的要求
        if box is not None:# 如果提供了框提示
            box = self.transform.apply_boxes(box, self.original_size) # 对输入的框坐标进行变换，使其与模型输入的图像尺寸匹配
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)# 将变换后的框坐标转换为 PyTorch 张量，并放置在与模型相同的设备上
            box_torch = box_torch[None, :] # 在第 0 维添加一个维度，以符合批量输入的要求
        if mask_input is not None: # 如果提供了低分辨率掩码输入
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)# 将低分辨率掩码输入转换为 PyTorch 张量，并放置在与模型相同的设备上
            mask_input_torch = mask_input_torch[None, :, :, :]# 在第 0 维添加一个维度，以符合批量输入的要求

        masks, iou_predictions, low_res_masks = self.predict_torch( # 调用 predict_torch 方法进行掩码预测
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].detach().cpu().numpy() # 将预测得到的掩码从 PyTorch 张量转换为 NumPy 数组，并移动到 CPU 上
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy() # 将预测的掩码质量分数从 PyTorch 张量转换为 NumPy 数组，并移动到 CPU 上
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy() # 将低分辨率的对数概率从 PyTorch 张量转换为 NumPy 数组，并移动到 CPU 上
        return masks_np, iou_predictions_np, low_res_masks_np  # 返回转换后的掩码、掩码质量分数和低分辨率对数概率的 NumPy 数组

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor], # 输入到模型的点提示的 PyTorch 张量，形状为 BxNx2，B 是批次大小，N 是点的数量，每个点以像素为单位表示为 (X, Y) 坐标，可为 None
        point_labels: Optional[torch.Tensor],  # 点提示的标签的 PyTorch 张量，形状为 BxN，1 表示前景点，0 表示背景点，可为 None
        boxes: Optional[torch.Tensor] = None, # 输入到模型的框提示的 PyTorch 张量，形状为 Bx4，格式为 XYXY（左上角和右下角坐标），可为 None
        mask_input: Optional[torch.Tensor] = None, # 输入到模型的低分辨率掩码的 PyTorch 张量，形状为 Bx1xHxW，对于 SAM 模型，H = W = 256，通常来自上一次预测迭代，可为 None
        multimask_output: bool = True, # 是否让模型返回三个掩码。对于模糊的输入提示（如单个点击），返回多个掩码通常能得到更好的结果；
        # 对于明确的提示（如多个输入提示），设置为 False 可能得到更好的结果
        return_logits: bool = False, # 是否返回未经过阈值处理的掩码对数概率，而不是二进制掩码
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.
        # 使用当前设置的图像，为给定的输入提示预测掩码。输入提示是批量的 PyTorch 张量，并且期望已经使用 ResizeLongestSide 转换到输入帧。

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
            #输出的掩码的 PyTorch 张量，格式为 BxCxHxW，其中 C 是掩码的数量，(H, W) 是原始图像的尺寸
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
            #形状为 BxC 的 PyTorch 张量，包含模型对每个掩码质量的预测
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
             #形状为 BxCxHxW 的 PyTorch 张量，其中 C 是掩码的数量，H = W = 256。这些低分辨率的对数概率可以作为掩码输入
        """
        if not self.is_image_set: # 检查是否已经设置了图像，如果未设置则抛出运行时错误
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if point_coords is not None:# 如果提供了点提示，则将点坐标和标签组合成一个元组
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
         #返回当前设置图像的图像嵌入，形状为 1xCxHxW，其中 C 是嵌入维度，(H, W) 是 SAM 模型的嵌入空间维度（通常 C = 256，H = W = 64）。
        """
        if not self.is_image_set:# 检查是否已经设置了图像，如果未设置则抛出运行时错误
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set." # 确保图像嵌入存在
        return self.features# 返回图像嵌入

    @property
    def device(self) -> torch.device:
        return self.model.device # 获取模型所在的设备

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False# 将图像设置状态标记为未设置
        self.features = None# 清空图像嵌入
        self.orig_h = None# 清空原始图像的高度
        self.orig_w = None # 清空原始图像的宽度
        self.input_h = None# 清空输入图像的高度
        self.input_w = None # 清空输入图像的宽度
