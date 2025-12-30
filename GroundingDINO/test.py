import torch
import collections


def print_parameter_count(model_dict, prefix=''):
    total_params = 0
    for name, param in model_dict.items():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(param, torch.Tensor):
            # 计算当前参数的数量
            num_params = param.numel()
            print(f"参数名: {full_name}, 参数量: {num_params}")
            total_params += num_params
        elif isinstance(param, collections.OrderedDict):
            # 递归处理嵌套字典
            nested_params = print_parameter_count(param, prefix=full_name)
            total_params += nested_params
    return total_params


# 加载 .pth 文件
model_dict = torch.load(r'C:\Users\yangxinyao\Desktop\GroundingDINO\3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799.pth', map_location=torch.device('cpu'))
# 打印各部分参数量
total = print_parameter_count(model_dict)
print(f"总参数量: {total}")