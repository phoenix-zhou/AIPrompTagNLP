import torch
from tqdm import tqdm
import time
from datetime import timedelta
import os
import pickle as pkl # 模型保存和导入

# 特殊符号
UNK, PAD, CLS = "[UNK]", "[PAD]", "[CLS]"  # padding符号, bert中综合信息符号
# 词表长度
MAX_VOCAB_SIZE = 10000  # 词表长度限制


def build_vocab(file_path, tokenizer, max_size, min_freq):
    """
    构建词汇表的函数。

    参数：
    - file_path (str): 包含文本数据的文件路径。
    - tokenizer (function): 用于分词的函数，接受一个字符串并返回分词后的结果。
    - max_size (int): 词汇表的最大大小，即保留的词汇数量上限。
    - min_freq (int): 词汇表中词语的最小出现频率，低于此频率的词汇将被过滤掉。
    """
    vocab_dic = {}  # 用于存储词汇表的字典，键为单词，值为单词出现的次数: vocab_dic={'hello': 5, 'world': 6}
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in tqdm(f):
            line = line.strip() # 去除两边空白
            if not line:
                continue
            content = line.split("\t")[0]  # 以制表符分隔的文本，这里取第一列的内容: line: content label
            # 使用给定的分词器（tokenizer）对文本进行分词，并更新词汇表
            for word in tokenizer(content):
                # .get(word, 0) 表示如果vocab_dic中存在word词对应的value, 则使用该value, 如果不存在, 则为0
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        # 根据词频对词汇表进行排序，并选择出现频率较高的词汇
        """
        vocab_dic.items(): 字典解包: 元组格式
        key: 排序依据,  reverse=True 表示降序
        key=lambda x: x[1] 说明:
            通过vocab_dic.items()解包vocab_dic, 解包结果如: x = ('hello', 5)
            x[1] 则等于 5, 结果按照vocab_dic的 value进行排序
        """
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq],
                            key=lambda x: x[1], reverse=True)[:max_size]
        # 将选定的词汇构建为字典，键为单词，值为索引
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        # 添加特殊符号到词汇表，例如未知符号（UNK）、填充符号（PAD）
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic


def build_dataset_CNN(config):
    # 定义字符级别分词器
    tokenizer = lambda x: [y for y in x]

    # 检查是否存在词汇表文件，如果存在则加载，否则构建新的词汇表
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, "rb"))
    else:
        # 构建词汇表
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        # 保存词汇表
        pkl.dump(vocab, open(config.vocab_path, "wb"))
    print(f"Vocab size: {len(vocab)}")

    # 定义加载数据集的辅助函数, pad_size 表示序列长度
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, "r", encoding="UTF-8") as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split("\t")
                words_line = []
                token = tokenizer(content) # token_id
                seq_len = len(token)

                # 填充或截断序列至指定长度
                if pad_size:
                    if len(token) < pad_size: # 用PAD补全
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size

                # 将词转换为对应的id
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                # 将数据添加到 contents 列表
                contents.append((words_line, int(label), seq_len))
        return contents  # [([...], 0), ([...], 1), ...]

    # 加载训练、验证和测试数据集
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)

    return vocab, train, dev, test


def build_dataset(config):
    """
    根据配置信息构建模型训练所需的数据集。
    参数：
    - config (object): 配置信息对象，包含有关数据集和模型的相关参数。
    返回：
    - train, dev, test (tuple): 包含三个元组，分别是训练集、验证集和测试集。
    """

    def load_dataset(path, pad_size=32):
        """
        加载并处理单个数据集文件。
        参数：
        - path (str): 数据集文件路径。
        - pad_size (int): 填充到的序列长度，默认为32。
        返回：
        - contents (list): 包含处理后的数据的列表。
        """
        contents = []  # 用于存储处理后的数据的列表
        with open(path, "r", encoding="UTF-8") as f:
            for line in tqdm(f):  # 逐行遍历文件内容
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split("\t")  # 分割每行的内容和标签
                token = config.tokenizer.tokenize(content)  # 使用分词器对内容进行分词
                token = [CLS] + token  # 在分词结果前加入[CLS]标记
                seq_len = len(token)  # 计算序列长度
                mask = []  # 用于存储填充掩码
                token_ids = config.tokenizer.convert_tokens_to_ids(token)  # 将分词结果转换为词汇索引
                if pad_size:
                    if len(token) < pad_size:  # 如果序列长度小于设定的填充长度
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))  # 创建填充掩码
                        token_ids += [0] * (pad_size - len(token))  # 对词汇索引列表进行填充
                    else:
                        mask = [1] * pad_size  # 如果序列长度大于等于填充长度，填充掩码为全1
                        token_ids = token_ids[:pad_size]  # 对词汇索引列表进行截断
                        seq_len = pad_size  # 更新序列长度为填充长度
                # 将处理后的数据添加到contents列表
                contents.append((token_ids, int(label), seq_len, mask))
        return contents

    # 使用load_dataset函数加载训练集、验证集和测试集
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    # 返回三个元组，分别是训练集、验证集和测试集
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device, model_name):
        """
        数据集迭代器的初始化函数。
        参数：
        - batches (list): 包含样本的列表。
        - batch_size (int): 每个批次的大小。
        - device (str): 数据加载到的设备（CPU或GPU）。
        - model_name (str): 使用的模型名称。
        """
        self.batch_size = batch_size  # 每个批次的大小
        self.batches = batches  # 包含样本的列表
        self.model_name = model_name  # 使用的模型名称
        self.n_batches = len(batches) // batch_size  # 批次的数量
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0  # 当前批次的索引
        self.device = device  # 数据加载到的设备（CPU或GPU）

    def _to_tensor(self, datas):
        """
        将样本数据转换为Tensor。
        """
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # 若为BERT模型，返回(x, seq_len, mask), y；
        if self.model_name == "bert":
            mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
            return (x, seq_len, mask), y
        # 若为TextCNN模型，返回(x, seq_len), y
        if self.model_name == "textCNN":
            return (x, seq_len), y

    def __next__(self):
        """
        获取下一个批次的样本。
        """
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        """
        返回迭代器对象本身。
        """
        return self

    def __len__(self):
        """
        获取迭代器的长度。
        """
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    """
    根据配置信息构建数据集迭代器。
    """
    iter = DatasetIterater(dataset, config.batch_size, config.device, config.model_name)
    return iter


def get_time_dif(start_time):
    """
    计算已使用的时间差。
    """
    # 获取当前时间
    end_time = time.time()
    # 计算时间差
    time_dif = end_time - start_time
    # 将时间差转换为整数秒，并返回时间差对象
    return timedelta(seconds=int(round(time_dif)))
