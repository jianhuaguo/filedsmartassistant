import os, cv2, torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms
import re

IMG_SIZE   = 224
BATCH_SIZE = 32
TRAIN_PATH = 'dataset/Train'
TEST_PATH  = 'dataset/Test'




def build_df(root):
    """
    解析文件夹名：以 'fresh' 或 'rotten' 开头，后面紧跟着水果名
    例：freshapples  -> fresh, apples
        rottentomato -> rotten, tomato
    """
    files, fruits, fresh = [], [], []
    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue
        # 用正则把前缀和水果名分开
        m = re.match(r'^(fresh|rotten)(.+)$', cls, flags=re.IGNORECASE)
        if not m:
            continue
        status, fruit = m.group(1).lower(), m.group(2).lower()
        # 去掉可能的拼写错误
        fruit = fruit.replace('tamto', 'tomato').replace('patato', 'potato')
        for img in os.listdir(cls_path):
            if img.lower().endswith(('.png', '.jpg', '.jpeg')):
                files.append(os.path.join(cls_path, img))
                fruits.append(fruit)
                fresh.append(0 if status == 'fresh' else 1)
    return pd.DataFrame({'path': files, 'fruit': fruits, 'fresh': fresh})

class FruitDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        img = cv2.imread(self.df.loc[idx, 'path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        fruit = torch.tensor(self.df.loc[idx, 'fruit_label'], dtype=torch.long)
        fresh = torch.tensor(self.df.loc[idx, 'fresh'], dtype=torch.long)
        return img, fruit, fresh

# 全局 transform
train_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def get_loaders():
    # 1. 合并 Train/Test（只是为了拿到全部类别）
    df_all  = pd.concat([build_df(TRAIN_PATH), build_df(TEST_PATH)])
    le = LabelEncoder()
    df_all['fruit_label'] = le.fit_transform(df_all['fruit'])
    # 2. 只拿原始 Train 做划分
    df_train = build_df(TRAIN_PATH)
    df_train['fruit_label'] = le.transform(df_train['fruit'])
    # 3. 训练/验证 8:2
    train_df, val_df = train_test_split(df_train, test_size=0.2,
                                        stratify=df_train[['fruit','fresh']],
                                        random_state=42)
    # 4. 测试集
    test_df = build_df(TEST_PATH)
    test_df['fruit_label'] = le.transform(test_df['fruit'])

    train_ds = FruitDataset(train_df, train_tf)
    val_ds   = FruitDataset(val_df,   val_tf)
    test_ds  = FruitDataset(test_df,  val_tf)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader, le
_all = pd.concat([build_df(TRAIN_PATH), build_df(TEST_PATH)])
le = LabelEncoder()
le.fit(_all['fruit'])