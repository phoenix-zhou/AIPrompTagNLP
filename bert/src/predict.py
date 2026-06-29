import torch
from importlib import import_module
import numpy as np

# 定义一些特殊的符号和类别映射
CLS = "[CLS]"
# 分类
id_to_name = {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science',
              5: 'society', 6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'}

def inference(model, config, input_text, pad_size=32):
    """
    模型推理函数，用于对输入文本进行情感分析的推理。
    参数：
    - model: 已加载的模型。
    - config: 模型配置信息。
    - input_text: 待分析的文本。
    - pad_size: 指定文本填充的长度。
    """
    # 1.分词(Token): 对输入文本进行分词和预处理
    content = config.tokenizer.tokenize(input_text)
    # 2. CLS + Token
    content = [CLS] + content
    seq_len = len(content)
    # 3. Token_ids
    token_ids = config.tokenizer.convert_tokens_to_ids(content)
    # 4. 输入数据格式: [token_ids, seq_len, Mask]
    # Mask 短补齐, 长截断: 填充或截断文本至指定长度
    if seq_len < pad_size:
        mask = [1] * len(token_ids) + [0] * (pad_size - seq_len)
        token_ids += [0] * (pad_size - seq_len)
    else:
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        seq_len = pad_size

    # 5. 前向传播(推理): to Tensor => 加batch_size
    # 将处理后的文本转换为PyTorch Tensor
    x = torch.LongTensor(token_ids).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)
    mask = torch.LongTensor(mask).to(config.device)

    # 增加一维，表示batch_size为1
    x = x.unsqueeze(0)
    seq_len = seq_len.unsqueeze(0)
    mask = mask.unsqueeze(0)
    data = (x, seq_len, mask) # 输入数据格式: [token_ids, seq_len, Mask]

    # 6.模型推理
    output = model(data) # 前向推理

    # 7.获取模型预测结果id
    predict_result = torch.max(output.data, 1)[1] # 获取分类: 0~9
    return predict_result


if __name__ == '__main__':
    # 加载BERT模型配置和模型
    model_name = 'bert'
    x = import_module('models.' + model_name)
    config = x.Config()

    # 设置随机种子，保证实验的可重复性
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    # 创建并加载BERT模型
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))

    # 待分析的文本
    input_text = '中国奥运女子单排夺冠: 国家体育局表示庆贺'

    # 进行模型推理
    res = inference(model, config, input_text)

    # 获取类别名
    result = id_to_name[res.item()]
    print(result)
