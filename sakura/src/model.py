import torch, torch.nn as nn, torchvision.models as models
class MultiTaskModel(nn.Module):
    def __init__(self, num_fruits=9):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
  # 2. 冻结所有参数（小数据集可全部冻结，大数据集可解冻最后两块）
        for p in self.backbone.parameters():
            p.requires_grad = False
        # 3. 去掉原 FC
        self.backbone.fc = nn.Identity()
        in_features = 2048   # ResNet50 输出特征维度

        # 4. 两个任务头
        self.head_fruit = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_fruits)
        )
        self.head_fresh = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        feat = self.backbone(x)
        return self.head_fruit(feat), self.head_fresh(feat)