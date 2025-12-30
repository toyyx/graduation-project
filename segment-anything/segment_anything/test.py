import torch


def count_parameters(pth_file_path):
    try:
        # 加载 .pth 文件
        state_dict = torch.load(pth_file_path,map_location=torch.device('cpu'))
        total_params = 0

        # 遍历 state_dict 中的所有参数
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                # 计算参数数量
                param_num = param.numel()
                print(f"参数名称: {name}, 参数数量: {param_num}")
                total_params += param_num

        return total_params
    except FileNotFoundError:
        print("错误: 文件未找到!")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")
    return None


if __name__ == "__main__":
    # 请替换为你的 .pth 文件的实际路径
    pth_file_path = './checkpoint_new/sam_conbine_epoch_88_5lr5.pth'
    param_count = count_parameters(pth_file_path)
    if param_count is not None:
        print(f"该 .pth 文件中的总参数量为: {param_count}")
    
    
    
