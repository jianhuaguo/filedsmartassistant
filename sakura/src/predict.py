import cv2, torch
from utils import val_tf
from model import MultiTaskModel
from sklearn.preprocessing import LabelEncoder
import os

def isFresh(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiTaskModel().to(device)
    model.load_state_dict(torch.load('best.pt', map_location=device, weights_only=True))
    model.eval()

    # 换成你的图片
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    tensor = val_tf(img).unsqueeze(0).to(device)

    class_names =  "apples",
    "banana",
    "bittergroud",
    "capsicum",
    "cucumber",
    "okra",
    "oranges",
    "potato",
    "tomato"
    le = LabelEncoder()
    le.fit(class_names)
    with torch.no_grad():
        fruit_logits, fresh_logits = model(tensor)
        fruit = le.inverse_transform([fruit_logits.argmax().item()])[0]
        fresh = 'Fresh' if fresh_logits.argmax().item()==0 else 'Spoiled'
        return fruit, fresh

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to test image")
    args = parser.parse_args()

    if not os.path.isfile(args.img):
        print("❌ 图片路径不存在")
    else:
        fruit, fresh = isFresh(args.img)
   
        print(f"Predicted => Fruit: {fruit}, Freshness: {fresh}")