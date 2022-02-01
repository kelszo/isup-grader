import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Enet(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.enet = EfficientNet.from_pretrained('efficientnet-b0', num_classes=out_dim)
        self.enet._avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.enet(x)
        return x
