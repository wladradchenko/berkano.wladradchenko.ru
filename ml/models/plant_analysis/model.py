import torch.nn as nn
from torchvision.models import mobilenet_v3_large

# ================= MODEL =================
class LeafNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_large(weights="DEFAULT")
        self.encoder = backbone.features

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.count_head = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        self.age_head = nn.Sequential(
            nn.Linear(960, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        feat = self.encoder(x)
        pooled = self.pool(feat).flatten(1)
        count = self.count_head(pooled)
        age = self.age_head(pooled)
        return feat, count, age

