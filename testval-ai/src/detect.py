import os
import argparse
from ultralytics import YOLO
import cv2
from utils import load_classes

from ultralytics.utils import LOGGER
import logging

LOGGER.setLevel(logging.ERROR)  # 只显示错误


def detect_pest(model_path, img_path, output_dir, conf=0.5, device='0'):
    """
    使用YOLOv8模型检测图像中的害虫
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 加载类别名称
    classes = load_classes()
    
    # 执行检测
    results = model.predict(
        img_path,
        conf=conf,
        device=device,
        save=True,
        project=os.path.dirname(output_dir),  # results
        name=os.path.basename(output_dir),    # uuid 名字
        show=False,
        line_width=2,
        verbose=False
    )
    
    # 解析检测结果
    detection_results = []
    for result in results:
        # 获取图像文件名
        img_name = os.path.basename(result.path)
        
        # 处理每个检测到的目标
        for box in result.boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # 获取类别和置信度
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0]) * 100
            
            # 获取类别名称
            if cls_id < len(classes):
                cls_name = classes[cls_id]
            else:
                cls_name = f"未知类别({cls_id})"
            
            detection_results.append({
                'image': img_name,
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2)
            })
    
    # print(f"检测完成，结果保存至: {output_dir}")
    return detection_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8害虫检测推理')
    parser.add_argument('--model', type=str, default='../weights/best.pt', help='模型权重路径')
    parser.add_argument('--img', type=str, required=True, help='输入图像路径或目录')
    parser.add_argument('--output', type=str, default='runs/detect', help='输出目录')
    parser.add_argument('--conf', type=float, default=0.5, help='置信度阈值')
    parser.add_argument('--device', type=str, default='0', help='推理设备')
    
    args = parser.parse_args()
    
    # 执行检测
    results = detect_pest(
        model_path=args.model,
        img_path=args.img,
        output_dir=args.output,
        conf=args.conf,
        device=args.device
    )
    
    # 打印检测结果
    print("\n检测结果:")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['class_name']} - 置信度: {res['confidence']:.2f}% - 位置: {[round(x) for x in res['bbox']]}")
