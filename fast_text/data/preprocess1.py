import jieba
# 存储数据集中的类别信息：id:label
id_to_label = {}
# id从0开始
idx = 0
# 打开数据集类别文件
with open('./class.txt', 'r', encoding='utf-8') as f1:
    # 遍历每一行文本
    for line in f1.readlines():
        # 去掉换行符和空白符
        line = line.strip('\n').strip()
        # 记录在字典中
        id_to_label[idx] = line
        # id增加
        idx += 1

print(f'id_to_label:{id_to_label}')

# 用来存储训练集数据
train_data = []
# 打开数据集文件，进行处理
with open('./train.txt', 'r', encoding='utf-8') as f2:
    # 获取每一行数据
    for line in f2.readlines():
        # 去掉换行符和空白符
        line = line.strip('\n').strip()
        # 获取文本和对应的标签信息
        sentence, label = line.split('\t')
        # 1.首先处理标签部分：获取标签id,并获取标签名称
        label_id = int(label)
        label_name = id_to_label[label_id]
        # 构建fasttext需要的训练集数据
        # fasttext标签
        new_label = '__label__' + label_name # 字符级别
        # 2.然后处理文本部分, 可以按字划分, 也可以按词划分
        # 这里是按照字符级别划分
        # sent_char = ' '.join(list(sentence)) # [中华女子] => [中, 华, 女, 子]
        # 这里是按照单词级别划分
        sent_char = ' '.join(jieba.lcut(sentence)) # [中华女子] => [中华, 女子]
        # 3.将文本和标签组合成fasttext规定的格式
        new_sentence = new_label + ' ' + sent_char # 字符级别划分 __label__education 中 华 女 子; 单词级别划分  __label__education 中华 女子
        # 4.将数据添加到list中
        train_data.append(new_sentence)
        # break

print(train_data[:5])

# 将数据处理后的结果存储在train_fast.txt文本中
with open('./train_fast1.txt', 'w', encoding='utf-8') as f3:
    # 遍历每一行数据
    for data in train_data:
        # 写入到文件中
        f3.write(data + '\n')
print('FastText训练数据预处理完毕!')