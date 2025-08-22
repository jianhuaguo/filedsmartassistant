import argparse
import os
from datetime import datetime
from ultralytics import YOLO

def evaluate_model(model_path, data_config, device='0', save_report=True):
    """
    评估YOLOv8模型性能并生成详细报告
    """
    # 加载模型
    model = YOLO(model_path)
    
    # 在测试集上评估
    metrics = model.val(
        data=data_config,
        split='test',  # 评估测试集
        device=device,
        save_json=True,  # 保存评估结果为JSON
        save=True,       # 保存可视化结果
        conf=0.001,      # 评估时的置信度阈值
        iou=0.6,         # IOU阈值
        plots=True       # 生成评估图表
    )
    
    # 打印评估报告标题
    print("\n" + "="*50)
    print(f"模型评估报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # 打印模型和数据集信息
    print(f"\n【评估信息】")
    print(f"模型路径: {os.path.abspath(model_path)}")
    print(f"数据集配置: {os.path.abspath(data_config)}")
    print(f"评估设备: {device}")
    print(f"评估结果保存路径: {metrics.save_dir}")
    
    # 打印关键指标（带中文解释）
    print("\n【核心评估指标】")
    print(f"{'mAP@0.5:':<15} {metrics.box.map50:.4f} (IoU=0.5时的平均精度，越高表示模型在中等阈值下检测效果越好)")
    print(f"{'mAP@0.5:0.95:':<15} {metrics.box.map:.4f} (IoU从0.5到0.95的平均精度，综合衡量不同阈值下的检测效果)")
    print(f"{'精确率:':<15} {metrics.box.mp:.4f} (预测为正例的样本中实际为正例的比例，越高表示误检越少)")
    print(f"{'召回率:':<15} {metrics.box.mr:.4f} (实际为正例的样本中被正确预测的比例，越高表示漏检越少)")
    
    # 计算并打印F1分数
    if len(metrics.box.f1) > 0:
        mean_f1 = sum(metrics.box.f1) / len(metrics.box.f1)
        print(f"{'平均F1分数:':<15} {mean_f1:.4f} (精确率和召回率的调和平均，综合反映模型性能)")
    else:
        print(f"{'平均F1分数:':<15} 无有效数据")
    
    # 打印单类别的评估指标（如果有多个类别）
    if hasattr(metrics.box, 'classes') and len(metrics.box.classes) > 1:
        print("\n【各类别评估指标】")
        print(f"{'类别ID':<8} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'mAP@0.5':<10}")
        print("-"*48)
        for i, cls in enumerate(metrics.box.classes):
            print(f"{cls:<8} {metrics.box.p[i]:<10.4f} {metrics.box.r[i]:<10.4f} {metrics.box.f1[i]:<10.4f} {metrics.box.map50[i]:<10.4f}")
    
    # 保存评估报告到文件
    if save_report:
        report_path = os.path.join(metrics.save_dir, "evaluation_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"模型评估报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
            f.write("【评估信息】\n")
            f.write(f"模型路径: {os.path.abspath(model_path)}\n")
            f.write(f"数据集配置: {os.path.abspath(data_config)}\n")
            f.write(f"评估设备: {device}\n")
            f.write(f"评估结果保存路径: {metrics.save_dir}\n\n")
            
            f.write("【核心评估指标】\n")
            f.write(f"mAP@0.5: {metrics.box.map50:.4f}\n")
            f.write(f"mAP@0.5:0.95: {metrics.box.map:.4f}\n")
            f.write(f"精确率: {metrics.box.mp:.4f}\n")
            f.write(f"召回率: {metrics.box.mr:.4f}\n")
            f.write(f"平均F1分数: {mean_f1:.4f}\n\n")
            
            if hasattr(metrics.box, 'classes') and len(metrics.box.classes) > 1:
                f.write("【各类别评估指标】\n")
                f.write(f"类别ID    精确率      召回率      F1分数      mAP@0.5\n")
                f.write("-"*48 + "\n")
                for i, cls in enumerate(metrics.box.classes):
                    f.write(f"{cls:<8} {metrics.box.p[i]:<10.4f} {metrics.box.r[i]:<10.4f} {metrics.box.f1[i]:<10.4f} {metrics.box.map50[i]:<10.4f}\n")
        
        print(f"\n【报告保存】评估报告已保存至: {report_path}")
    
    print("\n" + "="*50)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8害虫检测模型评估')
    parser.add_argument('--model', type=str, default='weights/best.pt', help='模型权重路径')
    parser.add_argument('--data', type=str, default='dataset.yaml', help='数据集配置文件路径')
    parser.add_argument('--device', type=str, default='0', help='评估设备，如"0"表示GPU，"cpu"表示CPU')
    parser.add_argument('--no-save', action='store_true', help='不保存评估报告')
    
    args = parser.parse_args()
    
    # 执行评估
    evaluate_model(
        model_path=args.model,
        data_config=args.data,
        device=args.device,
        save_report=not args.no_save
    )