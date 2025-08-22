import torch
from PIL import Image
from torchvision import transforms
import os

# --------------------------------------------------
# 1. 模型结构（假设你用 get_net() 定义在 models/model.py）
# --------------------------------------------------
from models.model import get_net   # 根据你的项目路径调整

# --------------------------------------------------
# 2. 加载类别映射
# --------------------------------------------------
def load_class_map(txt_path="class.txt"):
    mapping = {}
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            idx, name = line.strip().split("\t", 1)
            mapping[int(idx)] = name
    return mapping

CLASS_MAP = load_class_map()

# --------------------------------------------------
# 3. 初始化模型并加载权重
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = get_net().to(device)

ckpt_path = "checkpoints/best_model/resnet50/0/model_best.pth.tar"
checkpoint = torch.load(ckpt_path, map_location=device,weights_only=True)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# --------------------------------------------------
# 4. 图像预处理
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------------------------------------
# 5. 推理函数
# --------------------------------------------------
@torch.no_grad()
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    logits = model(tensor)
    pred_id = int(torch.argmax(logits, dim=1))
    return CLASS_MAP[pred_id]

# --------------------------------------------------
# 6. 命令行直接运行
# --------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to test image")
    args = parser.parse_args()

    if not os.path.isfile(args.img):
        print("❌ 图片路径不存在")
    else:
        result = predict(args.img)
        print("预测结果：", result)