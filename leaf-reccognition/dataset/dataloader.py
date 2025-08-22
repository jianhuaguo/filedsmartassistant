from torch.utils.data import Dataset
from torchvision import transforms as T 
from PIL import Image 
from itertools import chain 
from glob import glob
from tqdm import tqdm
import random 
import numpy as np 
import pandas as pd 
import os 
import cv2
import torch 
from config import config


#1.set random seed
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

#2.define dataset
class ChaojieDataset(Dataset):
    def __init__(self,label_list,transforms=None,train=True,test=False):
        self.test = test 
        self.train = train 
        imgs = []
        if self.test:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"]))
            self.imgs = imgs 
        else:
            for index,row in label_list.iterrows():
                imgs.append((row["filename"],row["label"]))
            self.imgs = imgs
        if transforms is None:
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485,0.456,0.406],
                                std = [0.229,0.224,0.225])])
            else:
                self.transforms  = T.Compose([
                    T.Resize((config.img_weight,config.img_height)),
                    T.RandomRotation(30),
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.RandomAffine(45),
                    T.ToTensor(),
                    T.Normalize(mean = [0.485,0.456,0.406],
                                std = [0.229,0.224,0.225])])
        else:
            self.transforms = transforms
    def __getitem__(self,index):
        if self.test:
            filename = self.imgs[index]
            img = Image.open(filename)
            img = self.transforms(img)
            return img,filename
        else:
            filename,label = self.imgs[index] 
            img = Image.open(filename)
            img = self.transforms(img)
            return img,label
    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    imgs, labels = zip(*batch)                 # 解包
    imgs = torch.stack(imgs, 0)                # float tensor
    labels = torch.tensor(labels, dtype=torch.long)  # long tensor
    return imgs, labels

def get_files(root, mode):
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(os.path.join(root, img))
        return pd.DataFrame({"filename": files})

    elif mode in ["train", "val"]:
        all_data_path, labels = [], []
        for label_name in os.listdir(root):
            label_path = os.path.join(root, label_name)
            if not os.path.isdir(label_path):
                continue
            try:
                label = int(label_name)
            except ValueError:
                continue  # 跳过非数字文件夹

            for ext in ["*.jpg", "*.JPG"]:
                for img_path in glob(os.path.join(label_path, ext)):
                    all_data_path.append(img_path)
                    labels.append(label)

        return pd.DataFrame({"filename": all_data_path, "label": labels})

    else:
        print("check the mode please!")
    
