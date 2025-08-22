import os
import shutil
import random
from sklearn.model_selection import train_test_split

def split_dataset(img_dir, label_dir, output_dir, train_ratio=0.7, val_ratio=0.2):
    """
    划分数据集为训练集、验证集和测试集
    """
    # 创建输出目录结构
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # 获取所有图片文件
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(img_extensions)]
    random.shuffle(img_files)
    total = len(img_files)
    
    if total == 0:
        print("未找到图片文件!")
        return
    
    # 划分数据集
    train_files, temp_files = train_test_split(img_files, test_size=1-train_ratio, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=1-(val_ratio/(1-train_ratio)), random_state=42)
    
    # 复制文件函数
    def copy_files(file_list, split_name):
        for file in file_list:
            # 复制图片
            src_img = os.path.join(img_dir, file)
            dst_img = os.path.join(output_dir, split_name, 'images', file)
            shutil.copy(src_img, dst_img)
            
            # 复制标签
            label_file = os.path.splitext(file)[0] + '.txt'
            src_label = os.path.join(label_dir, label_file)
            if os.path.exists(src_label):
                dst_label = os.path.join(output_dir, split_name, 'labels', label_file)
                shutil.copy(src_label, dst_label)
    
    # 复制各数据集文件
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    # 打印划分结果
    print(f"总样本数: {total}")
    print(f"训练集: {len(train_files)} ({len(train_files)/total:.2%})")
    print(f"验证集: {len(val_files)} ({len(val_files)/total:.2%})")
    print(f"测试集: {len(test_files)} ({len(test_files)/total:.2%})")
    print("数据集划分完成!")

if __name__ == "__main__":
    # 配置路径
    IMG_DIR = "data/images"
    LABEL_DIR = "datasets/temp/labels"  # 从convert_xml.py生成的标签
    OUTPUT_DIR = "datasets"
    
    # 执行划分
    split_dataset(IMG_DIR, LABEL_DIR, OUTPUT_DIR)
