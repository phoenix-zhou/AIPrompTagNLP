# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from transformers.optimization import AdamW
from tqdm import tqdm


def loss_fn(outputs, labels):
    """
    定义损失函数，使用交叉熵损失。
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def train(config, model, train_iter, dev_iter):
    """
    模型训练函数。
    参数：
    - config: 配置信息对象。
    - model: 待训练的模型。
    - train_iter: 训练集的迭代器。
    - dev_iter: 验证集的迭代器。
    """
    # 记录开始训练的时间
    start_time = time.time()

    # 权重衰减策略, 目的: 防止过拟合
    # 参数优化器设置
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01
         },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0
         }]

    # 1.设置优化器
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    dev_best_loss = float("inf")
    # 2.定义损失函数loss_fn()

    # 将模型设置为训练模型
    model.train()
    # 3.两个遍历: 遍历每个轮次, 遍历每个批次
    # 遍历每个轮次
    for epoch in range(config.num_epochs):
        total_batch = 0
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        # 遍历每个批次
        for i, (trains, labels) in enumerate(tqdm(train_iter)):
            # (1). 模型前向传播
            outputs = model(trains)
            # (2). 梯度清零
            model.zero_grad()
            # (3). 计算损失
            loss = loss_fn(outputs, labels)
            # (4). 反向传播
            loss.backward()
            # (5). 参数更新
            optimizer.step()

            # 每100个batch输出在训练集和验证集上的效果
            if total_batch % 100 == 0 and total_batch != 0:
                true = labels.data.cpu()
                """
                [0.2, 0.33, 0.69, ...] => [0.69, 3]
                [0.25, 0.73, 0.58, ...] => [0.73, 2]
                [0.12, 0.323, 0.863, ...] => [0.863, 5]
                [batch_size, num_classes]
                """
                predic = torch.max(outputs.data, 1)[1].cpu() # 预测值: 获取最大概率数据
                train_acc = metrics.accuracy_score(true, predic) # acc评估
                # 评估验证集效果
                dev_acc, dev_loss = evaluate(config, model, dev_iter) # 计算损失
                # 若验证集损失更低，保存模型参数
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path) # 保存最优模型
                    improve = "*"
                else:
                    improve = ""
                # 计算时间差
                time_dif = get_time_dif(start_time)
                # 输出训练和验证集上的效果
                # > 表示右对齐
                msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                # 评估完成后将模型置于训练模式, 更新参数
                model.train()
            # 每个batch结束后累加计数
            total_batch += 1


def test(config, model, test_iter):
    """
    模型测试函数，用于在测试集上进行最终的模型测试。
    参数：
    - config: 配置信息对象。
    - model: 待测试的模型。
    - test_iter: 测试集的数据迭代器。
    """
    # 采用量化模型进行推理时需要关闭
    # model.eval()

    start_time = time.time()
    # 调用验证函数计算评估指标
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True) # test=True 开启Test模式

    # 打印测试结果信息:输出测试集上的损失、准确率、分类报告和混淆矩阵等信息
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time) # get_time_dif方法 在utils.py中
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    """
    模型评估函数。
    参数：
    - config: 配置信息对象。
    - model: 待评估的模型。
    - data_iter: 数据迭代器。
    - test: 是否为测试集评估。
    """

    # 采用量化模型进行推理时需要关闭
    # model.eval()

    loss_total = 0
    # 预测结果
    predict_all = np.array([], dtype=int)
    # label信息
    labels_all = np.array([], dtype=int)
    # 不进行梯度计算
    with torch.no_grad():
        # 遍历数据集
        for texts, labels in data_iter:
            # 将数据送入网络中
            outputs = model(texts)
            # 损失函数
            loss = F.cross_entropy(outputs, labels)
            # 损失和
            loss_total += loss
            # 获取label信息
            labels = labels.data.cpu().numpy()
            # 获取预测结果
            predic = torch.max(outputs.data, 1)[1].cpu().numpy() # [batch_size, num_classes]
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    # 计算准确率
    acc = metrics.accuracy_score(labels_all, predict_all) # Acc计算

    if test: # 测试评估
        # 如果是测试集评估，计算分类报告和混淆矩阵
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all) # 混淆矩阵 bascase
        return acc, loss_total / len(data_iter), report, confusion
    else:
        # 如果是验证集评估，仅返回准确率和平均损失
        return acc, loss_total / len(data_iter)
