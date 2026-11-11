import torch
import torch.nn as nn

class WideBasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size= 3,stride=stride ,padding= 1 , bias =False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size= 3,padding= 1 , bias=False),

        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size= 1,stride=stride , bias=False),
            )

    def forward(self,x):
        return self.shortcut(x) + self.feature_extractor(x)




class WideResNet28(nn.Module):
    ...