import requests
url = "http://127.0.0.1:8000/predict"
files = {'file': open(rf'dataset\Train\freshokra\o_f001.png', 'rb')}
r = requests.post(url, files=files)
print(r.status_code)   # 先看状态码
print(r.text)     