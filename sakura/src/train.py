import torch, torch.nn as nn, torchmetrics, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from src.model import MultiTaskModel
from src.utils import get_loaders

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 10
    ALPHA  = 0.7   # loss = α*fruit + (1-α)*fresh

    train_loader, val_loader, test_loader, le = get_loaders()

    model = MultiTaskModel().to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss()

    fruit_acc = torchmetrics.Accuracy(task='multiclass', num_classes=9).to(device)
    fresh_acc = torchmetrics.Accuracy(task='multiclass', num_classes=2).to(device)

    history = {'train_loss':[], 'val_loss':[],
            'train_fruit':[], 'train_fresh':[],
            'val_fruit':[], 'val_fresh':[]}

    def run_epoch(loader, training=False):
        if training:
            model.train()
        else:
            model.eval()
        epoch_loss, fruit_true, fresh_true = 0., [], []
        fruit_pred, fresh_pred = [], []
        for imgs, fruit, fresh in tqdm(loader, leave=False):
            imgs, fruit, fresh = imgs.to(device), fruit.to(device), fresh.to(device)
            if training:
                opt.zero_grad()
            with torch.set_grad_enabled(training):
                out_fruit, out_fresh = model(imgs)
                loss = ALPHA*loss_fn(out_fruit, fruit) + (1-ALPHA)*loss_fn(out_fresh, fresh)
                if training:
                    loss.backward()
                    opt.step()
            epoch_loss += loss.item() * imgs.size(0)
            fruit_true.extend(fruit.cpu().tolist())
            fresh_true.extend(fresh.cpu().tolist())
            fruit_pred.extend(out_fruit.argmax(1).cpu().tolist())
            fresh_pred.extend(out_fresh.argmax(1).cpu().tolist())
        epoch_loss /= len(loader.dataset)
        fruit_score = fruit_acc(torch.tensor(fruit_pred), torch.tensor(fruit_true)).item()
        fresh_score = fresh_acc(torch.tensor(fresh_pred), torch.tensor(fresh_true)).item()
        return epoch_loss, fruit_score, fresh_score

    best_val = 0.
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_fruit, tr_fresh = run_epoch(train_loader, training=True)
        val_loss, val_fruit, val_fresh = run_epoch(val_loader)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_fruit'].append(tr_fruit)
        history['train_fresh'].append(tr_fresh)
        history['val_fruit'].append(val_fruit)
        history['val_fresh'].append(val_fresh)
        print(f"Epoch {epoch:02d} | "
            f"tr_loss={tr_loss:.3f} tr_fruit={tr_fruit:.3f} tr_fresh={tr_fresh:.3f} | "
            f"val_loss={val_loss:.3f} val_fruit={val_fruit:.3f} val_fresh={val_fresh:.3f}")
        # 保存最佳
        score = (val_fruit + val_fresh)/2
        if score > best_val:
            best_val = score
            torch.save(model.state_dict(), 'best.pt')

    # 画曲线
    plt.style.use('seaborn-v0_8')
    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    plt.title('Loss'); plt.plot(history['train_loss'],label='train'); plt.plot(history['val_loss'],label='val'); plt.legend()
    plt.subplot(1,3,2)
    plt.title('Fruit Acc'); plt.plot(history['train_fruit'],label='train'); plt.plot(history['val_fruit'],label='val'); plt.legend()
    plt.subplot(1,3,3)
    plt.title('Fresh Acc'); plt.plot(history['train_fresh'],label='train'); plt.plot(history['val_fresh'],label='val'); plt.legend()
    plt.tight_layout(); plt.savefig('curve.png'); plt.show()

    # 在测试集评估
    model.load_state_dict(torch.load('best.pt'))
    test_loss, test_fruit, test_fresh = run_epoch(test_loader)
    print(f"Test  | fruit_acc={test_fruit:.3f} fresh_acc={test_fresh:.3f}")
if __name__ == '__main__':
    # Windows 必须加这一行
    torch.multiprocessing.freeze_support()
    main()