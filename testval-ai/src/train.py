import argparse
from ultralytics import YOLO
import os
import shutil
import torch

def train_model(
    data_config, 
    model_path,  # 改为已有模型路径
    epochs,  # 增量训练轮次可适当减少
    img_size, 
    batch_size, 
    device='0',
    save_path='pest_detection_dataaugment'
):

    # 1. 如果有，则加载已训练的模型
    if model_path and model_path.strip() != '':
        model = YOLO(model_path)  # 加载已有模型，如weights/best_optimized.pt
    else:
        model = YOLO("yolo11.yaml")  # 加载自定义训练模型
    
    # 2. 检查GPU是否可用
    if device != 'cpu' and not torch.cuda.is_available():
        print("警告：未检测到GPU，自动切换到CPU训练")
        device = 'cpu'
    
    # 3. 训练参数配置（关键调整）
    results = model.train(
        # 基础配置
        data=data_config,  # 指定数据集配置文件（YAML格式），包含训练和验证路径、类别名等
        epochs=epochs,  # 总训练轮数
        imgsz=img_size,  # 输入图片的尺寸（例如640表示640x640）
        batch=batch_size,  # 每批训练图像数量
        device=device,  # 训练使用的设备（如 '0' 表示GPU 0，'cpu' 表示CPU）

        project='runs/train',  # 训练结果保存的主目录
        name=save_path,  # 当前训练的子目录名称，防止覆盖之前的实验
        pretrained=True,  # 是否加载官方预训练模型（False 表示使用自己的模型权重）

        # # 4. 数据增强保持不变（继续学习多样本）
        augment=True,  # 是否启用数据增强
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.2,
        hsv_h=0.005,
        hsv_s=0.5,
        hsv_v=0.2,

        # 5. 损失函数权重可微调（针对弱项优化）
        box=12.0,  # 边界框回归损失权重（对定位错误更敏感）
        cls=0.5,  # 分类损失权重（用于提升分类性能）
        dfl=2.0,  # 分布式Focal Loss权重（提升边界框精度）

        # 6. 优化器与学习率策略（关键：降低学习率，避免破坏已有参数）
        optimizer='AdamW',  # 优化器类型（AdamW 适合微调与稳定训练）
        lr0=0.0005,  # 初始学习率（设为首次训练的一半，保护已有知识）
        lrf=0.01,  # 最低学习率相对于初始学习率的倍数（用于cos调度）
        cos_lr=True,  # 使用cosine学习率衰减策略（逐步降低学习率）
        warmup_epochs=5,  # 热身训练轮数（初期缓慢提高学习率）

        # 7. 正则化与早停（保持模型稳定）
        weight_decay=0.0005,  # 权重衰减（L2正则项，防止过拟合）
        dropout=0.1,  # Dropout比率（防止过拟合，随机屏蔽部分神经元）
        patience=15,  # 早停的耐心轮数（若验证集精度15轮未提升则提前停止）

        # 8. 验证与可视化
        val=True,  # 启用验证过程（每轮结束后评估验证集性能）
        plots=True,  # 绘制训练过程中的可视化图表（如loss、mAP变化）
        save=True,  # 是否保存训练过程中最好的模型
    )

    # 保存增量训练后的最佳模型（区分原模型）
    os.makedirs('weights', exist_ok=True)
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        print(f"增量训练完成! 新最佳模型已保存至: weights/best_incremental.pt")
    else:
        print(f"训练完成，但未找到最佳模型文件: {best_model_path}")
    
    print(f"增量训练分析图表路径: {results.save_dir}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv11害虫检测模型（增量训练版）')
    parser.add_argument('--data', type=str, default='dataset.yaml', help='数据集配置路径')
    parser.add_argument('--model', type=str, default="",
                      help='已有模型路径（如weights/best_optimized.pt）')
    parser.add_argument('--epochs', type=int, default=100, help='增量训练轮次（建议30-80）')
    parser.add_argument('--img-size', type=int, default=960, help='输入尺寸（与原模型一致）')
    parser.add_argument('--batch', type=int, default=16, help='批次大小（与原模型一致）')
    parser.add_argument('--device', type=str, default='0', help='设备（0/GPU编号或cpu）')
    parser.add_argument('--save-path', type=str, default="", help='保存路径（默认为runs/train/）')
    
    args = parser.parse_args()
    
    # 执行增量训练
    train_model(
        data_config=args.data,
        model_path=args.model,  # 传入已有模型路径
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch,
        device=args.device,
        save_path=args.save_path
    )
