def get_key_points_from_doodle(mask):
    """
    从二值掩码中提取关键点（端点、拐点、均匀采样点）
    参数：
        mask: 二值化后的涂鸦掩码（0-1矩阵）
    返回：
        points: 关键点坐标列表[[x,y],...]
    """
    # 轮廓检测（参考交通标线识别逻辑[9]）
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for cnt in contours:
        # 端点检测（获取首尾点）
        start_point = tuple(cnt[0][0])
        end_point = tuple(cnt[-1][0])
        points.extend([start_point, end_point])

        # 拐点检测（使用多边形近似法[3]）
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) > 2:
            points.extend([tuple(p[0]) for p in approx])

        # 计算轮廓的长度和面积
        length = cv2.arcLength(cnt, False)
        area = cv2.contourArea(cnt)

        # 根据长度和面积确定采样间隔
        # 这里使用一个简单的规则，你可以根据实际情况调整
        if length < 100 and area < 100:
            step = max(1, len(cnt) // 2)  # 长度和面积都较小时，每1/2长度采样
        elif length < 200 and area < 200:
            step = max(1, len(cnt) // 3)  # 长度和面积适中时，每1/3长度采样
        else:
            step = max(1, len(cnt) // 5)  # 长度和面积较大时，每1/5长度采样

        # 均匀采样中间点（类似速度分析采样[2]）
        for i in range(0, len(cnt), step):
            points.append(tuple(cnt[i][0]))

    return np.unique(points, axis=0)  # 去重