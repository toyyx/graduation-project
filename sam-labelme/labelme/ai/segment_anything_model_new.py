import collections
import threading

import imgviz
import numpy as np
import onnxruntime
import skimage
import traceback

import torch

from .sam import sam_model_registry
from .sam.utils.transforms import ResizeLongestSide
from ..logger import logger
from . import _utils


class SegmentAnythingModel_new:
    def __init__(self, pth_path):
        self.pth_path = pth_path
        self._model = self.init_model(pth_path)
        self._transform = ResizeLongestSide(self._model.image_encoder.img_size)
        self._image_size = 1024

        self._lock = threading.Lock()
        self._image_embedding_cache = collections.OrderedDict()
        self._image_embedding_event = threading.Event()
        self._thread = None

    def init_model(self,checkpoint_path=None):
        model_type = "vit_b"

        # 加载SAM模型
        sam = sam_model_registry[model_type]()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sam.to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)
        log = sam.load_state_dict(checkpoint, strict=False)
        print("Model loaded from {} \n => {}".format(checkpoint_path, log))
        return sam

    def set_image(self, image: np.ndarray):
        with self._lock:
            self._image = image[:, :, :3]#hwc
            self._image_embedding = self._image_embedding_cache.get(
                self._image.tobytes()
            )

        if self._image_embedding is None:
            self._thread = threading.Thread(
                target=self._compute_and_cache_image_embedding
            )
            self._thread.start()
            self._image_embedding_event.wait()

    def _compute_and_cache_image_embedding(self):
        with self._lock:
            logger.debug("Computing image embedding...")

            self._image_embedding = self._compute_image_embedding()
            if len(self._image_embedding_cache) > 10:
                self._image_embedding_cache.popitem(last=False)
            self._image_embedding_cache[self._image.tobytes()] = self._image_embedding
            logger.debug("Done computing image embedding.")
        self._image_embedding_event.set()


    def _get_image_embedding(self):
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            return self._image_embedding

    def predict_mask_from_points(self, points, point_labels):
        return self._compute_mask_from_points(
            points=points,
            point_labels=point_labels,
        )

    def predict_polygon_from_points(self, points, point_labels):
        mask = self.predict_mask_from_points(points=points, point_labels=point_labels)
        return mask,_utils.compute_polygon_from_mask(mask=mask)



    def _compute_image_embedding(self):
        # image = self._image.permute(2, 0, 1)#chw
        input_image, target_size = self._transform.apply_image_return_target_size(self._image)  # 应用 ResizeLongestSide 变换，将图像的最长边调整为模型期望的尺寸
        input_image_torch = torch.as_tensor(input_image, device=self._model.device)  # 将 NumPy 数组转换为 PyTorch 张量，并将其放置在与模型相同的设备上
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :,:]  # 调整张量的维度顺序为 BCHW（批次、通道、高度、宽度），并确保内存连续

        self.original_size = self._image.shape[:2]  # 保存原始图像的尺寸
        self.input_size = tuple(input_image_torch.shape[-2:])  # 保存输入图像（变换后）的尺寸
        input_image = self._model.preprocess(input_image_torch)  # 对输入图像进行预处理，如归一化、填充等操作
        image_embeddings = self._model.image_encoder(input_image)

        return image_embeddings

    def adjust_point_to_bbox(self,bbox):
        x1,y1,x2,y2 = bbox
        new_x1=min(x1,x2)
        new_y1=min(y1,y2)
        new_x2=max(x1,x2)
        new_y2=max(y1,y2)

        return [new_x1,new_y1,new_x2,new_y2]

    def _compute_mask_from_points(self, points, point_labels):
        input_point = np.array(points, dtype=np.float32)  # 点
        input_label = np.array(point_labels, dtype=np.int32)
        assert len(input_point) == len(input_label)

        if len(input_label) == 2 and input_label[0] == 2 and input_label[1] == 3:  # bbox
            # 假设 points 是 [[x1, y1], [x2, y2]] 格式的数组
            # 提取坐标并转为列表 [x1, y1, x2, y2]
            bbox = input_point.flatten().tolist()  # 4
            bbox=self.adjust_point_to_bbox(bbox)
            bbox=np.array(bbox)[None,:] #1,4
            #bbox_tensor = torch.as_tensor(bbox)
            bbox_tensor = torch.tensor(self._transform.apply_boxes(bbox, self.original_size),device=self._model.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配

            sparse_embeddings, dense_embeddings = self._model.prompt_encoder(
                points=None,
                boxes=bbox_tensor,
                boxes_origin=None,
                rboxes=None,
                rboxes_origin=None,
                masks=None
            )
            low_res_masks, iou_predictions = self._model.mask_decoder(
                image_embeddings=self._get_image_embedding(),
                image_pe=self._model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False  ##
            )
            masks = self._model.postprocess_masks(  # 对低分辨率掩码进行后处理，将其调整到输入图像的原始尺寸
                low_res_masks,
                input_size=self.input_size,  ##
                original_size=self.original_size,
            )#1 1 hw

            mask = masks[0, 0]  # (1, 1, H, W) -> (H, W)
            mask = (mask > 0.0).cpu().numpy()

            MIN_SIZE_RATIO = 0.05
            skimage.morphology.remove_small_objects(
                mask, min_size=mask.sum() * MIN_SIZE_RATIO, out=mask
            )

            if 0:
                imgviz.io.imsave("mask.jpg", imgviz.label2rgb(mask, imgviz.rgb2gray(image)))
            return mask


        elif len(input_label) == 4 and input_label[0] == 4 and input_label[1] == 5 and input_label[1] == 6 and input_label[1] == 7:  # rbox
            rbox_np = input_point#4,2
            rbox_np = np.array(sort_rbox_points(rbox_np)) #4,2
            bbox_np = np.array(calculate_bbox(rbox_np))[None,:]  #1,4
            rbox_np = rbox_np[None,:] # 1,4,2

            bbox_tensor = torch.tensor(self._transform.apply_boxes(bbox_np, self.original_size),device=self._model.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配
            rbox_tensor = torch.tensor(self._transform.apply_rboxes(rbox_np, self.original_size), device=self._model.device)  # 对输入的点坐标进行变换，使其与模型输入的图像尺寸匹配
            sparse_embeddings, dense_embeddings = self._model.prompt_encoder(
                points=None,
                boxes=bbox_tensor,
                boxes_origin=None,
                rboxes=rbox_tensor,
                rboxes_origin=None,
                masks=None
            )
            low_res_masks, iou_predictions = self._model.mask_decoder(
                image_embeddings=self._get_image_embedding(),
                image_pe=self._model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False  ##
            )
            masks = self._model.postprocess_masks(  # 对低分辨率掩码进行后处理，将其调整到输入图像的原始尺寸
                low_res_masks,
                input_size=self.input_size,  ##
                original_size=self.original_size,
            )  # 1 1 hw

            mask = masks[0, 0]  # (1, 1, H, W) -> (H, W)
            mask = (mask > 0.0).cpu().numpy()

            MIN_SIZE_RATIO = 0.05
            skimage.morphology.remove_small_objects(
                mask, min_size=mask.sum() * MIN_SIZE_RATIO, out=mask
            )

            if 0:
                imgviz.io.imsave("mask.jpg", imgviz.label2rgb(mask, imgviz.rgb2gray(image)))
            return mask
        else:
            points_tensor = torch.tensor(self._transform.apply_coords(input_point, self.original_size),device=self._model.device)[None,:]
            label_tensor= torch.tensor(input_label,device=self._model.device)[None,:]
            sparse_embeddings, dense_embeddings = self._model.prompt_encoder(
                points=(points_tensor,label_tensor),
                boxes=None,
                boxes_origin=None,
                rboxes=None,
                rboxes_origin=None,
                masks=None
            )
            low_res_masks, iou_predictions= self._model.mask_decoder(
                image_embeddings=self._get_image_embedding(),
                image_pe=self._model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False  ##
            )
            masks = self._model.postprocess_masks(  # 对低分辨率掩码进行后处理，将其调整到输入图像的原始尺寸
                low_res_masks,
                input_size=self.input_size,  ##
                original_size=self.original_size,
            )  # 1 1 hw

            mask = masks[0, 0]  # (1, 1, H, W) -> (H, W)
            mask = (mask > 0.0).cpu().numpy()

            MIN_SIZE_RATIO = 0.05
            skimage.morphology.remove_small_objects(
                mask, min_size=mask.sum() * MIN_SIZE_RATIO, out=mask
            )

            if 0:
                imgviz.io.imsave("mask.jpg", imgviz.label2rgb(mask, imgviz.rgb2gray(image)))
            return mask


"""
def _compute_scale_to_resize_image(image_size, image):
    height, width = image.shape[:2]
    if width > height:
        scale = image_size / width
        new_height = int(round(height * scale))
        new_width = image_size
    else:
        scale = image_size / height
        new_height = image_size
        new_width = int(round(width * scale))
    return scale, new_height, new_width


def _resize_image(image_size, image):
    scale, new_height, new_width = _compute_scale_to_resize_image(
        image_size=image_size, image=image
    )
    scaled_image = imgviz.resize(
        image,
        height=new_height,
        width=new_width,
        backend="pillow",
    ).astype(np.float32)
    return scale, scaled_image
"""


def sort_rbox_points(rbox):
    """
    按 top_point、right_point、bottom_point、left_point 的顺序排序 rbox 的点
    :param rbox: 旋转包围框的四个点坐标
    :return: 排序后的旋转包围框的四个点坐标
    """
    rbox = np.array(rbox)
    x_values = rbox[:, 0]
    y_values = rbox[:, 1]
    # 判断 x 或 y 值是否都不相同
    if len(set(x_values)) == 4 and len(set(y_values)) == 4:
        top_point = rbox[np.argmin(rbox[:, 1])]
        right_point = rbox[np.argmax(rbox[:, 0])]
        bottom_point = rbox[np.argmax(rbox[:, 1])]
        left_point = rbox[np.argmin(rbox[:, 0])]
        return [top_point.tolist(), right_point.tolist(), bottom_point.tolist(), left_point.tolist()]
    else:
        # 手动设定矩形
        min_x = np.min(x_values)
        max_x = np.max(x_values)
        min_y = np.min(y_values)
        max_y = np.max(y_values)

        top_point = [min_x, min_y]
        right_point = [max_x, min_y]
        bottom_point = [max_x, max_y]
        left_point = [min_x, max_y]
        return [top_point, right_point, bottom_point, left_point]

def calculate_bbox(rbox):
    """
    计算最小水平外接框、偏移比值和面积比值
    :param rbox: 旋转包围框的四个点坐标
    :return: new_box, fix, ratio
    """
    rbox = np.array(rbox)
    x_min = np.min(rbox[:, 0])
    y_min = np.min(rbox[:, 1])
    x_max = np.max(rbox[:, 0])
    y_max = np.max(rbox[:, 1])
    new_bbox_xyxy = [x_min, y_min, x_max, y_max]
    return new_bbox_xyxy






