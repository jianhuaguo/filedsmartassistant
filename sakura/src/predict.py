import cv2, torch
from src.utils import val_tf, le
from src.model import MultiTaskModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MultiTaskModel().to(device)
model.load_state_dict(torch.load('best.pt', map_location=device, weights_only=True))
model.eval()

path = fr'dataset\Train\freshcapsicum\capsicum3_0.jpg_0_1542.jpg'          # 换成你的图片
img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
tensor = val_tf(img).unsqueeze(0).to(device)

with torch.no_grad():
    fruit_logits, fresh_logits = model(tensor)
    fruit = le.inverse_transform([fruit_logits.argmax().item()])[0]
    fresh = 'Fresh' if fresh_logits.argmax().item()==0 else 'Spoiled'

print(f"Predicted => Fruit: {fruit}, Freshness: {fresh}")