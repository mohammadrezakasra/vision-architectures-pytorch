import torch
import torch.nn as nn

class ResNextBaseBlock(nn.Module):
    def __init__(self, in_channels,out_channels,kernel_size,stride):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size = 3,stride=stride,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size = 3,padding=1),
        )

        self.shortcut = nn.Sequential()
        if stride !=1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size =1,stride=stride),
            )
    def forward(self, x):
        return self.shortcut(x) + self.feature_extractor(x)


class ResNext29(nn.Module):
    ...
