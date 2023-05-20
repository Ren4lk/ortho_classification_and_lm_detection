from torch import nn
from torchvision import models


class Network(nn.Module):
    def __init__(self, num_classes=78):  # 39*2
        super().__init__()

        self.model_name = 'resnet50'
        self.model = models.resnet50()
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
