# 1.导入依赖包
import fasttext

# 指定训练集和测试集数据
train_data_path = './data/train_fast.txt'
test_data_path = './data/test_fast.txt'

# 2.开启模型训练
# 获取训练模型(有监督训练) -> 得到训练好的模型对象
model = fasttext.train_supervised(input=train_data_path, wordNgrams=2)
print(f'词的数量: {len(model.words)}')  # 词的数量: 4760
"""
标签: ['__label__stocks', '__label__science', '__label__society', '__label__game', '__label__sports',
 '__label__finance', '__label__politics', '__label__education', '__label__realty',
  '__label__entertainment']
"""
print(f'标签: {model.labels}')

# 3.开启模型测试
result = model.test(test_data_path)
# 输出测试结果
print(f'resutl:{result}')
