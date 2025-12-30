import imgviz
import numpy as np
import skimage
import cv2

from labelme.logger import logger


def _get_contour_length(contour):
    contour_start = contour
    contour_end = np.r_[contour[1:], contour[0:1]]
    return np.linalg.norm(contour_end - contour_start, axis=1).sum()


def compute_polygon_from_mask(mask):
    if mask is not None:
        # 将布尔型 mask 转换为 uint8 类型
        mask = (mask * 255).astype(np.uint8)
        # 找到mask的边界
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 逼近多边形
        approx_level = 0.001
        epsilon = approx_level * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)

        # 限制多边形的顶点数量，调整近似精度
        num_vertices = 50
        points_num = approx.shape[0]
        iter = 0
        modulation = 2  # 每次epsilon调整的大小
        while points_num < num_vertices or points_num > int(1.5 * num_vertices):
            if points_num < num_vertices:
                approx_level = approx_level / (modulation / 1.2)
            else:
                approx_level = approx_level * modulation
            epsilon = approx_level * cv2.arcLength(contours[0], True)
            approx = cv2.approxPolyDP(contours[0], epsilon, True)
            points_num = approx.shape[0]
            iter = iter + 1
            if iter >20:
                break

        # print("多边形顶点坐标数量：", approx.shape[0])
        # print("多边形顶点坐标：")
        polygon = []
        for point in approx:
            x, y = point.ravel()
            point_ = [int(x), int(y)]
            polygon.append(point_)
        # print(polygon)

        return polygon
    else:
        return None


