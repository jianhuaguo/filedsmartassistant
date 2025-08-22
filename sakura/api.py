import os, uvicorn, cv2
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np 
from src.utils import val_tf, le
from src.model import MultiTaskModel
from fastapi.middleware.cors import CORSMiddleware



# ---------- 模型初始化 ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiTaskModel().to(device)
model.load_state_dict(torch.load('best.pt', map_location=device, weights_only=True))
model.eval()

app = FastAPI(title="Fruit-Freshness API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 调试阶段先全部放行
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. 读取图片
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 2. 预处理
    tensor = val_tf(img).unsqueeze(0).to(device)
    # 3. 推理
    with torch.no_grad():
        fruit_logits, fresh_logits = model(tensor)
        fruit = le.inverse_transform([fruit_logits.argmax().item()])[0]
        fresh = 'Fresh' if fresh_logits.argmax().item() == 0 else 'Spoiled'
    # 4. 返回 JSON
    return JSONResponse(content={"fruit": fruit, "freshness": fresh})
# ---------- 启动 ----------
# 本地 HTTPS（自签名证书） ----------
# 生成一次即可：
# openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )