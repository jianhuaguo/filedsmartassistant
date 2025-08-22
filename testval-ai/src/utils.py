import os
import cv2
import matplotlib.pyplot as plt
import shutil

def load_classes(classes_path='classes.txt'):
    """加载类别名称列表"""
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"类别文件不存在: {classes_path}")
    
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = [line.strip() for line in f.readlines()]
    
    return classes

def visualize_detections(image_path, detections, output_path=None):
    """可视化检测结果"""
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 绘制检测框和标签
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cls_name = det['class_name']
        conf = det['confidence']
        
        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 绘制标签
        label = f"{cls_name}: {conf:.1f}%"
        cv2.putText(
            image, 
            label, 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 0), 
            2
        )
    
    # 保存图像
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, image_bgr)
    
    return image

def plot_training_curves(results_path):
    """绘制训练曲线"""
    # 从results.csv加载数据并绘制损失和mAP曲线
    # 实际使用时可以扩展此函数
    pass

def clear_temp_files():
    """清理临时文件"""
    temp_dirs = ['runs', 'datasets/temp']
    for dir in temp_dirs:
        if os.path.exists(dir):
            shutil.rmtree(dir)
            print(f"已清理: {dir}")
