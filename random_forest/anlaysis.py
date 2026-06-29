"""
## 数据集分析  analysis.py
# 1.导入依赖包
# 2.读取数据
    # 2.1 打印前10行
    # 2.2 获取样本数量
    # 2.3 统计每个类别的数量
    # 2.4 打印信息
# 3.统计样本总量
    # 3.1 遍历每个类别的样本数量
    # 3.2 计算每个类别样本数量的比例
    # 3.3 绘制图表
    # 3.4 统计每行样本的长度
    # 3.5 统计文本长度的均值和方差
# 4.结巴分词
    # 4.1 添加一列数据存储分词后的结果
    # 4.2 分词
    # 4.3 将分词后的结果只保留30个元素
    # 4.4 结果存放到csv文件中
"""

# 1.导入依赖包
import matplotlib.pyplot as plt # 画图
import pandas as pd # pd.read_csv()导入数据
from collections import Counter # 计数
import numpy as np
import jieba # jieba分词

# 2.读取数据
content = pd.read_csv('./data/train.txt', sep='\t')

# 2.1 打印前10行
"""
sentence  label
0         中华女子学院：本科层次仅1专业招男生      3
1     两天价网站背后重重迷雾：做个网站究竟要多少钱      4
2  东5环海棠公社230-290平2居准现房98折优惠      1
3  卡佩罗：告诉你德国脚生猛的原因 不希望英德战踢点球      7
4    82岁老太为学生做饭扫地44年获授港大荣誉院士      5
5       记者回访地震中可乐男孩：将受邀赴美国参观      5
6          冯德伦徐若�隔空传情 默认其是女友      9
7     传郭晶晶欲落户香港战伦敦奥运 装修别墅当婚房      1
8           《赤壁OL》攻城战诸侯战硝烟又起      8
9                “手机钱包”亮相科博会      4
"""
print(content.head(10))
# 打印content列名字
"""
Index(['sentence', 'label'], dtype='str')
"""
print(content.columns)

# 2.2 获取样本数量
"""
180000
"""
print(len(content)) # 180000

# 2.3 统计每个类别的数量
count = Counter(content.label.values)

# 2.4 打印信息
"""
Counter({np.int64(3): 18000, np.int64(4): 18000, np.int64(1): 18000, np.int64(7): 18000, np.int64(5): 18000, np.int64(9): 18000, np.int64(8): 18000, np.int64(2): 18000, np.int64(6): 18000, np.int64(0): 18000})
"""
print(count)
"""
10
"""
print(len(count))

print('*'*20)

# 3.统计样本总量
values = [] # 每个样本的数量
ratios = [] # 每个样本的比例
total = 0 # 总的样本数量
# 3.1 遍历每个类别的样本数量
for i, v in count.items():
    total += v
    values.append(v)
print(f'total: {total}') # total: 180000
# 3.2 计算每个类别样本数量的比例
for i, v in count.items():
    ratio = v / total * 100
    ratios.append(ratio)
    """
    3 10.0 %
    4 10.0 %
    1 10.0 %
    7 10.0 %
    5 10.0 %
    9 10.0 %
    8 10.0 %
    2 10.0 %
    6 10.0 %
    0 10.0 %
    """
    print(i, ratio, "%")

# 3.3 绘制图表
# plt.bar(range(10), values)
# plt.show()
# plt.pie(ratios, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# plt.title('样本类别比例图')
# plt.show()
#
# print('*'*20)


# 3.4 统计每行样本的长度
content['sentence_len'] = content['sentence'].apply(len)
"""
sentence head:                    sentence  label  sentence_len
0         中华女子学院：本科层次仅1专业招男生      3            18
1     两天价网站背后重重迷雾：做个网站究竟要多少钱      4            22
2  东5环海棠公社230-290平2居准现房98折优惠      1            25
3  卡佩罗：告诉你德国脚生猛的原因 不希望英德战踢点球      7            25
4    82岁老太为学生做饭扫地44年获授港大荣誉院士      5            23
"""
print(f'sentence head:{content.head()}')
# 3.5 统计文本长度的均值和方差
length_mean = np.mean(content['sentence_len']) # 均值
length_std = np.std(content['sentence_len']) # 方差
print('length_mean = ', length_mean) # length_mean =  19.21257222222222
print('length_std = ', length_std) # length_std =  3.8637872533601523


# 4.结巴分词
def cut_sentence(s):
    return list(jieba.cut(s)) # 返回分词数据, 并转换成list

# 4.1 添加一列数据存储分词后的结果
content['words'] = content['sentence'].apply((cut_sentence))
# 打印前十行
"""
content words:                     sentence  ...                                              words
0         中华女子学院：本科层次仅1专业招男生  ...           [中华, 女子, 学院, ：, 本科, 层次, 仅, 1, 专业, 招, 男生]
1     两天价网站背后重重迷雾：做个网站究竟要多少钱  ...   [两天, 价, 网站, 背后, 重重, 迷雾, ：, 做个, 网站, 究竟, 要, 多少, 钱]
2  东5环海棠公社230-290平2居准现房98折优惠  ...  [东, 5, 环, 海棠, 公社, 230, -, 290, 平, 2, 居, 准现房, 9...
3  卡佩罗：告诉你德国脚生猛的原因 不希望英德战踢点球  ...  [卡佩罗, ：, 告诉, 你, 德国, 脚, 生猛, 的, 原因,  , 不, 希望, 英德...
4    82岁老太为学生做饭扫地44年获授港大荣誉院士  ...  [82, 岁, 老太, 为, 学生, 做饭, 扫地, 44, 年, 获授, 港大, 荣誉, 院士]

[5 rows x 4 columns]
"""
print(f'content: {content.head()}')
# 4.2 分词
"""
content words:                     sentence  ...                                     words
0         中华女子学院：本科层次仅1专业招男生  ...              中华 女子 学院 ： 本科 层次 仅 1 专业 招 男生
1     两天价网站背后重重迷雾：做个网站究竟要多少钱  ...        两天 价 网站 背后 重重 迷雾 ： 做个 网站 究竟 要 多少 钱
2  东5环海棠公社230-290平2居准现房98折优惠  ...   东 5 环 海棠 公社 230 - 290 平 2 居 准现房 98 折 优惠
3  卡佩罗：告诉你德国脚生猛的原因 不希望英德战踢点球  ...  卡佩罗 ： 告诉 你 德国 脚 生猛 的 原因   不 希望 英德 战 踢 点球
4    82岁老太为学生做饭扫地44年获授港大荣誉院士  ...       82 岁 老太 为 学生 做饭 扫地 44 年 获授 港大 荣誉 院士
"""
content['words'] = content['sentence'].apply(lambda s: ' '.join(cut_sentence(s)))
print(f'content: {content.head()}')

# 4.3 将分词后的结果只保留30个元素
# s.split() 去除每个words中的空格等,
# ' '.join(s.split()) 将 s.split() 的操作重新以 ' ' 拼接成字符串
# [:30] 截取处理后的字符串的前 30 个字符
# 为什么是截取前30个: 根据上面 均值length_mean =  19.21257222222222 得知: 截取数据 = 3 * 均值
content['words'] = content['words'].apply(lambda s: ' '.join(s.split())[:30])

# 4.4 结果存放到csv文件中
content.to_csv('./data/train_new.csv')