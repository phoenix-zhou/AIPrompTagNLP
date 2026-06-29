"""
# Bert模型训练
# 1.导入依赖包
# 2.导入对应模型的配置和模型定义
# 3.构建训练、验证、测试数据集和数据迭代器
# 4.创建模型实例并移至指定设备
# 5.模型训练
# 6.模型测试
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

if __name__ == "__main__":
    if args.model == "bert":
        # 2.导入对应模型的配置和模型定义
        model_name = "bert"
        x = import_module("models." + model_name) # 在 models/bert.py下: Config, Model
        config = x.Config() # 导入配置文件, 配置超参
        print(config.dev_path)

        # 设置随机种子，保证实验的可重复性
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        print("Loading data for Bert Model...")

        # 3.构建训练、验证、测试数据集和数据迭代器(构建可迭代数据集)
        train_data, dev_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)

        # 4.创建模型实例并移至指定设备(实例化模型)
        model = x.Model(config).to(config.device)

        # 5.训练模型(模型训练)
        train(config, model, dev_iter, dev_iter)

        # 6.在测试集上测试模型性能(模型测试)
        test(config, model, test_iter)
