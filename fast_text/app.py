"""
# Fasttext服务化app（服务端）
# 1.导入依赖包
# 2.实例化Flask对象
# 3.定义请求响应函数
# 4.启动Flask服务
"""

# 1.导入依赖包
import time
import jieba
import fasttext
import re

# 服务框架使用Flask, 导入工具包
from flask import Flask, Response, json
from flask import request

# 2.实例化Flask对象
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 解决中文乱码问题

# 加载自定义的停用词表
jieba.load_userdict('./data/stopwords.txt')

# 提供已经训练好的模型路径
model_save_path = './data/fasttext1_1780755816.bin'

# 实例化fasttext对象, 并加载模型参数用于推断, 提供服务请求
model = fasttext.load_model(model_save_path)
print('FastText模型实例化完毕...')

# 3.设定投满分项目的服务的路由和请求方法
@app.route('/main_server',  methods=["POST"])
def main_server():
    # # 接收来自请求方发送的服务字段(form表单格式)
    # uid = request.form['uid']
    # text = request.form['text']

    # 接收来自请求方发送的服务字段(json格式)
    request_json = request.get_json()
    uid = request_json['uid']
    content = request_json['content']
    # 调用推理函数获取预测结果
    t1 = time.time()

    # 对请求文本进行处理, 因为前面加载的是基于分词的模型, 所以这里也要对text进行分词操作
    input_text = ' '.join(jieba.lcut(content))
    # 执行模型的预测
    pred = model.predict(input_text)
    print(f'pred: {pred}')
    # 结果输出
    result = pred[0][0]
    # 对结果正则化
    match = re.search(r'__label__(.+)', result)
    result = match.group(1)

    # 定义响应数据格式
    t2 = time.time()
    respose_data = {
        'status': 'success',
        "result": result,
        'time': '{:.4f}s'.format(t2 - t1)
    }
    # 返回请求数据
    # sort_keys: 按照respose_data参数顺序返回,不然返回时参数的顺序会乱
    return Response(status=200, response=json.dumps(respose_data, sort_keys=False))

# 启动Flask服务
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)