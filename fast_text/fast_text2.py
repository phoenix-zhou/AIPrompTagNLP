"""
# Fasttext优化1
# 1.导入依赖包
# 2.模型训练
# 3.模型测试
# 4.模型保存
"""

# 1.导入依赖包
import fasttext
import time

# 2.开启模型训练
# 获取训练集,测试集和验证集的路径
train_data_path = './data/train_fast.txt'
dev_data_path = './data/dev_fast.txt'
test_data_path = './data/test_fast.txt'
# autotuneValidationFile 参数需要指定验证数据集所在的路径
# 它将在验证集是使用随机搜索的方法寻找最优的超参数
# 使用autotuneDuration参数可以控制随机搜索的时间, 默认是300秒.
# 根据不同的需求, 可以延长或者缩短时间.
# 调节的超参数包含这些内容:
# lr                         学习率 default 0.1
# dim                        词向量维度 default 100
# ws                         上下文窗口大小 default 5， cbow
# epoch                      epochs 数量 default 5
# minCount                   最低词频 default 5
# wordNgrams                 n-gram设置 default 1
# loss                       损失函数 {hs,softmax} default softmax
# minn                       最小字符长度 default 0
# maxn                       最大字符长度 default 0
# 设置verbose来观察超参数的值
# verbose: 该参数决定日志打印级别, 当设置为3, 可以将当前正在尝试的超参数打印出来
model = fasttext.train_supervised(input=train_data_path, # 训练集路径
                                autotuneValidationFile=dev_data_path, # 验证集路径
                                autotuneDuration=12,  # 时间单位为s
                                wordNgrams=2, # N_gram
                                verbose=3)

# 3.开启模型测试
result = model.test(test_data_path)
print(f'result: {result}')

# 4.模型保存
# 获取当前时间,作为模型文件的名称
time1 = int(time.time())
model_save_path = "./data/fasttext_{}.bin".format(time1)
model.save_model(model_save_path)
print(f'操作成功.')