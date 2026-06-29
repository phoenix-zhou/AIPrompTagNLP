import requests
import time

# 定义请求url和传入的data
url = "http://127.0.0.1:6000/v1/main_server"
data = {"uid": "AI-223265", "text": "中国奥运女子单排夺冠: 国家体育局表示庆贺"}

start_time = time.time()

# 向服务发送post请求
res = requests.post(url, data=data)
cost_time = time.time() - start_time

# 打印返回的结果
print('文本类别: ', res.text)
print('单条样本耗时: ', cost_time * 1000, 'ms')
