import torch, cv2, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import val_tf, get_loaders, le           # 复用之前写好的函数
from src.model import MultiTaskModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def main() -> None:


    # 1. 加载数据（测试集或验证集均可）
    _, _, test_loader, _ = get_loaders()

    # 2. 加载最优权重
    model = MultiTaskModel().to(device)
    ckpt = torch.load('best.pt', map_location=device)
    model.load_state_dict(ckpt)
    model.eval()

    # 3. 收集全部预测与真值
    fruit_true, fruit_pred = [], []
    fresh_true, fresh_pred = [], []

    with torch.no_grad():
        for imgs, y_fruit, y_fresh in test_loader:
            imgs = imgs.to(device)
            logits_fruit, logits_fresh = model(imgs)
            fruit_true.extend(y_fruit.cpu().numpy())
            fresh_true.extend(y_fresh.cpu().numpy())
            fruit_pred.extend(logits_fruit.argmax(1).cpu().numpy())
            fresh_pred.extend(logits_fresh.argmax(1).cpu().numpy())

    # 4. 整体指标
    print('\n========== Fruit Classification Report ==========')
    mask = np.isin(fruit_true, list(range(len(le.classes_))))
    fruit_true = np.array(fruit_true)[mask]
    fruit_pred = np.array(fruit_pred)[mask]

    print(classification_report(
            fruit_true, fruit_pred,
            labels=None,
            target_names=[le.classes_[i] for i in np.unique(fruit_true)],
            digits=3,
            zero_division=0))

    print('\n========== Freshness Classification Report ==========')
    print(classification_report(fresh_true, fresh_pred,
                                target_names=['Fresh', 'Spoiled'], digits=3))

    # 5. 混淆矩阵
    def plot_cm(y_true, y_pred, labels, title):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(title.replace(' ', '_') + '.png')
        plt.show()

    plot_cm(fruit_true, fruit_pred, le.classes_, 'Confusion Matrix - Fruit')
    plot_cm(fresh_true, fresh_pred, ['Fresh', 'Spoiled'], 'Confusion Matrix - Freshness')

    # 6. 保存预测结果 CSV（方便后续错误分析）
    df_out = pd.DataFrame({
        'fruit_true': le.inverse_transform(fruit_true),
        'fruit_pred': le.inverse_transform(fruit_pred),
        'fresh_true': np.where(np.array(fresh_true)==0, 'Fresh', 'Spoiled'),
        'fresh_pred': np.where(np.array(fresh_pred)==0, 'Fresh', 'Spoiled'),
    })
    df_out.to_csv('test_predictions.csv', index=False)
    print('\n预测结果已保存到 test_predictions.csv')


if __name__ == "__main__":
    main()