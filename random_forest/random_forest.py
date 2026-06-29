"""
## 数据集分析  random_forest.py
# 1.导入依赖包
# 2.读取数据集
# 3.构建语料库
# 4.获取停用词
# 5.计算tfidf特征stopwords.txt
# 6.划分数据集
# 7.实例化模型
# 8.模型训练
# 9.模型评估
"""

# 1.导入依赖包
from sklearn.feature_extraction.text import TfidfVectorizer # TF+ IDF
from sklearn.model_selection import train_test_split # 拆分数据集和测试集
from sklearn.ensemble import RandomForestClassifier  # 随机森林算法
import pandas as pd
# from icecream import ic # 美化版本print
from tqdm import tqdm # 进度条
from sklearn.metrics import accuracy_score # 准确率 Acc
from sklearn.metrics import recall_score  # 召回率
from sklearn.metrics import precision_score  # 精确率
from sklearn.metrics import f1_score  # f1_score

# 2.读取数据集
# 指定数据集的位置
TRAIN_CORPUS = './data/train_new.csv' # 经过anlaysis.py 数据集分析处理后的数据集
STOP_WORDS = './data/stopwords.txt'  # 停用词数据
WORDS_COLUMN = 'words' # TRAIN_CORPUS中的词: 中华 女子 学院 ： 本科 层次 仅 1 专业 招 男生

# 获取数据集
content = pd.read_csv(TRAIN_CORPUS)

# 3.构建语料库
corpus = content[WORDS_COLUMN]
"""
0      中华 女子 学院 ： 本科 层次 仅 1 专业 招 男生
1    两天 价 网站 背后 重重 迷雾 ： 做个 网站 究竟 要 
2    东 5 环 海棠 公社 230 - 290 平 2 居 准现
3    卡佩罗 ： 告诉 你 德国 脚 生猛 的 原因 不 希望 英
4    82 岁 老太 为 学生 做饭 扫地 44 年 获授 港大 
Name: words, dtype: str
"""
print(corpus.head())
corpus = corpus.values
"""
<ArrowStringArray>
[  '中华 女子 学院 ： 本科 层次 仅 1 专业 招 男生', '两天 价 网站 背后 重重 迷雾 ： 做个 网站 究竟 要 ',
 '东 5 环 海棠 公社 230 - 290 平 2 居 准现', '卡佩罗 ： 告诉 你 德国 脚 生猛 的 原因 不 希望 英',
 '82 岁 老太 为 学生 做饭 扫地 44 年 获授 港大 ', '记者 回访 地震 中 可乐 男孩 ： 将 受邀 赴美国 参观',
        '冯德伦 徐若 � 隔空 传情 默认 其是 女友', '传 郭晶晶 欲 落户 香港 战 伦敦 奥运 装修 别墅 当婚',
      '《 赤壁 OL 》 攻城战 诸侯 战 硝烟 又 起',               '“ 手机 钱包 ” 亮相 科博会',
 ...
     '印度 孟买 接到 恐怖 爆炸 恐吓 全城 加强 戒备',       '四六级 听力 冲刺 ： 八大 宝典 助 你 过关',
 'Q 杂志 2008 年 50 大 专辑 酷玩不敌 Kings', '张敬轩 王宛 之 音乐剧 三度 加场 朝夕 相对 感情 亲密',
      '美国 特使 敦促 阿富汗 总统 候选人 保持 克制', '侧滑 掌上 PSP 手机 索爱 Xperia Play 评测',
    '皖通 高速 跌 0.8% 国元 香港 给予 买入 评级',                '《 无限 江湖 》 近日 更新',
 '廊坊 十九 城邦 别墅 现 内部 认购 96 折 均价 70', '连续 上涨 7 个 月 糖价 7 月 可能 出现 调整 行情']
Length: 180000, dtype: str
"""
print(f'corpus: {corpus}')

# 4.获取停用词
# open: 以 UTF-8 编码格式打开停用词文件
# .read()：将文件中的所有内容一次性读取为一个完整的长字符串
# split()：按空白字符（空格、换行符 \n、制表符 \t 等）将这个长字符串切分成一个列表（List
stop_words = open(STOP_WORDS, encoding='utf-8').read().split() # split 按空白符分割并去除所有的空白符

# 5.计算 TF-IDF特征stopwords.txt
# TfidfVectorizer(stop_words=stop_words)：初始化一个 TF-IDF 向量化器，并将上一步得到的停用词列表传入。
# 这样在后续计算时，向量化器会自动过滤掉这些毫无意义的词（如“的”、“是”、“了”等），只统计有用的词, 从而降低特征维度并提高模型质量
tfidf = TfidfVectorizer(stop_words=stop_words)
"""
fit_transform(corpus)：这是核心操作，提取有用的特征, 得到 TF-IDF 矩阵, 分为两步：
    fit：学习语料库 corpus（通常是一个包含多篇文档的列表），构建词汇表（Vocabulary），并计算每个词的 IDF（逆文档频率）值。
    transform：将 corpus 中的每一篇文档，根据刚刚学到的词汇表和 IDF 值，转换成 TF-IDF 权重向量
"""
text_vectors = tfidf.fit_transform(corpus)
"""
print(tfidf.vocabulary_)：打印出模型学到的词汇表。
这是一个字典格式，键是单词，值是该单词在矩阵中的列索引（例如：{'在此': 37548, '黄埔区': 110122, ...}）
"""
# print(f'tfidf.vocabulary: {tfidf.vocabulary_}')
"""
text_vectors: <Compressed Sparse Row sparse matrix of dtype 'float64'
	with 1198491 stored elements and shape (180000, 110830)>
  Coords	Values
  (0, 13986)	0.42241145988168566
  (0, 41696)	0.2790551166361433
  (0, 43200)	0.3652773396659266
  (0, 65426)	0.33684764676878726
  (0, 46146)	0.5212140897922078
  (0, 12830)	0.296037829480461
  (0, 78673)	0.37081046985124755
  
  对上诉讲解: 第一篇文章(也就是第一行数据)的TF-IDF, 统计了7个, 也就是保留了7个单词的数据, 这7个单词是有用的数据
  (1, 13463)	0.33339969892652577
  (1, 86665)	0.5004012168578689
  (1, 88783)	0.2781898855532912
  (1, 102405)	0.36163033288954227
  (1, 99997)	0.3736224823130161
  (1, 21747)	0.420166965441128
  (1, 83045)	0.33785072842183655
  (2, 72407)	0.4239847881485823
  (2, 23516)	0.4336942741937884
  (2, 1315)	0.4078622204858592
  (2, 1621)	0.4359059529002595
  (2, 25414)	0.525148155549066
  (3, 30484)	0.40936540034346186
  (3, 34779)	0.44620719236454076
  (3, 51363)	0.3318062651989599
  (3, 78020)	0.5302309726872778
  (3, 31085)	0.36613211012168634
  (3, 47693)	0.3286732517249506
  :	:
  (179996, 85977)	0.3989713566658273
  (179996, 94625)	0.2899445088964127
  (179996, 109465)	0.3633522656188282
  (179996, 36953)	0.426618516937443
  (179996, 79653)	0.516988646290227
  (179997, 70509)	0.4656004601889316
  (179997, 63994)	0.46456058606951883
  (179997, 62043)	0.4929455909405545
  (179997, 99329)	0.5695650250101982
  (179998, 27115)	0.24839800040491453
  (179998, 3072)	0.2823659157232217
  (179998, 37789)	0.22547477787656361
  (179998, 3576)	0.329479011763212
  (179998, 94142)	0.32714933579797095
  (179998, 24457)	0.3464085021173238
  (179998, 49095)	0.36093858802066947
  (179998, 29152)	0.42817658009292603
  (179998, 38192)	0.3981648428314937
  (179999, 92671)	0.3604797966310664
  (179999, 32918)	0.334119646845884
  (179999, 95297)	0.35188907108886947
  (179999, 11828)	0.3467248819811138
  (179999, 25900)	0.37638968848549625
  (179999, 99814)	0.38087961267625514
  (179999, 84725)	0.4771119771344868
  
  总共180000行数据, 和train_new.csv的行数(文档数)对应
"""
# 打印出转换后的特征矩阵。由于文本数据通常很稀疏（大部分词在一篇文档中都不会出现），
# 这里输出的会是一个稀疏矩阵（Sparse Matrix），格式通常显示为 <行数 x 列数> 以及 <行, 列> 值 的形式，以节省内存
# print(f'text_vectors: {text_vectors}')

# 目标值
target = content['label']

# 6.划分数据集: text_vectors 传入的是经过特征工程后的向量以及label
x_train, x_test, y_train, y_test = train_test_split(text_vectors, target, test_size=0.2, random_state=30)
# 7.实例化模型
model = RandomForestClassifier()
# 8.模型训练
model.fit(x_train, y_train)
# 模型预测
y_pre = model.predict(x_test)

# 9.模型评估
acc = accuracy_score(y_pre, y_test)
print(f'准确率 acc: {acc}')
# print(f'精确率: {precision_score(y_test, y_pre)}')
# print(f'召回率: {recall_score(y_test, y_pre)}')
# print(f'F1值: {f1_score(y_test, y_pre)}')

print('结束')