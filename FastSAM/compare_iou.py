def extract_iou(line):
    """从行中提取 iou 值"""
    parts = line.split('iou:')
    if len(parts) > 1:
        try:
            return float(parts[1].strip())
        except ValueError:
            return None
    return None


def compare_iou_files(file1_path, file2_path):
    """对比两个文件中对应行的 iou 值，并统计各自更高的次数"""
    file1_higher_count = 0
    file1_higher_sum=0
    file2_higher_count = 0
    file2_higher_sum = 0
    equal_count = 0

    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            for line1, line2 in zip(file1, file2):
                iou1 = extract_iou(line1)
                iou2 = extract_iou(line2)

                if iou1 is not None and iou2 is not None:
                    if iou1 > iou2:
                        file1_higher_count += 1
                        file1_higher_sum += iou1-iou2
                    elif iou2 > iou1:
                        file2_higher_count += 1
                        file2_higher_sum += iou2-iou1
                    else:
                        equal_count += 1
                else:
                    print("文件未找到2，请检查文件路径。")
                    return

    except FileNotFoundError:
        print("文件未找到，请检查文件路径。")
        return

    print(f"文件 {file1_path} 中 iou 更高的次数: {file1_higher_count}  {file1_higher_sum}")
    print(f"文件 {file2_path} 中 iou 更高的次数: {file2_higher_count}  {file2_higher_sum}")
    print(f"两个文件中 iou 相等的次数: {equal_count}")


# 示例使用
file1_path = 'eva_coco_rbox.log'
file2_path = 'eva_coco_0.95.log'
compare_iou_files(file1_path, file2_path)