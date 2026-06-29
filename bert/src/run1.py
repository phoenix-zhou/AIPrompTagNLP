"""
## Bert模型量化
# 1.导入依赖包
# 2.导入对应模型的配置和模型定义
# 3.数据迭代器的预处理和生成
# 4.实例化模型并加载参数，注意不要加载到GPU上，只能在CPU上实现模型量化
# 5.量化BERT模型
# 6.测试量化后的模型在测试集上的表现
# 7.保存量化后的模型
"""

# 1.导入依赖包
import torch
import numpy as np
from train_eval import train, test # 训练, 验证函数
from importlib import import_module # 导入模块, 导入类
import argparse # 参数
from utils import build_dataset, build_iterator, get_time_dif

# 命令行参数解析
# 1. 定义解析器
parser = argparse.ArgumentParser(description="Chinese Text Classification")
# 2. 解析器调用入口
# 3. 添加参数
parser.add_argument("--model", type=str, default='bert', help="choose a model: bert")
args = parser.parse_args()

if __name__ == '__main__':
    # 2.导入对应模型的配置和模型定义
    if args.model == 'bert':
        # 2.导入对应模型的配置和模型定义
        model_name = "bert"
        x = import_module("models." + model_name)  # 在 models/bert.py下: Config, Model
        config = x.Config()  # 导入配置文件, 配置超参
        print(config.dev_path)

        # 设置随机种子，保证实验的可重复性
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        print("Loading data for Bert Model...")

        # 3.数据迭代器的预处理和生成(构建可迭代数据集)
        print('Loading data for Bert Model...')
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)

        # 4.实例化模型并加载参数，注意不要加载到GPU上，只能在CPU上实现模型量化(实例化模型)
        model = x.Model(config)
        print(model)
        model.load_state_dict(torch.load(config.save_path, map_location='cpu'))

        # 5.量化BERT模型
        quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print(quantized_model)

        # 6.测试量化后的模型在测试集上的表现
        test(config, quantized_model, test_iter)

        # 7.保存量化后的模型
        torch.save(quantized_model, config.save_path2)
