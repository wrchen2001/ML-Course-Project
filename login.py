
# coding=UTF-8

import requests
import time


url = 'http://10.9.1.3'
data = {
    "DDDDD": '20234227052',  # 这行是你需要根据自己的情况修改的地方
    "upass": 'CWRcwr756623',  # 这行是你需要根据自己的情况修改的地方
    "R1": "0",
    "R3": "1",
    "R6": "0",
    "pare": "00",
    "OMKKey": "123456",
}
header = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en,en-US;q=0.9,zh-CN;q=0.8,zh;q=0.7",
    "Cache-Control": "max-age=0",
    "Connectin": "keep-alive",
    "Host": "10.9.1.3",
    "Referer": "http://10.9.1.3/?isReback=1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36 Edg/90.0.818.56",
}
while True:
    time.sleep(5)
    response = requests.post(url, data, headers=header).status_code  # 获取状态码
    print("回应代码{}".format(response))  # 打印状态码

