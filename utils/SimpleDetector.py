import torch.nn as nn
import torchvision.models as models


class SimpleDetector(nn.Module):
    def __init__(self, num_classes, grid_size=13, pretrained=True):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.num_output = num_classes + 5  # objectness + cx,cy,w,h + classes

        backbone = models.resnet18(weights = models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # bo avgpool + fc

        # Detection head
        self.head = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),  # Dropout=0.3, tránh overfit

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.3),  # Dropout=0.3, tránh overfit

            nn.Conv2d(256, self.num_output, 1)
        )

        # Adaptive pooling de ra dung grid_size x grid_size
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.head(x)   # (B, 5+C, S, S)
        x = x.permute(0, 2, 3, 1).contiguous()   # (B, S, S, 5+C)

        return x
