from typing import Any

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
    def __init__(self,in_channels = 3 , wide_factor= 2,depth= 28,num_classes = 10):
        super().__init__()
        num_blocks = (depth - 4) // 6

        self.conv1 = nn.Conv2d(in_channels,16,kernel_size=3,stride=1,padding=1 ,bias=False)

        self.stage1 = self._make_layer(WideBasicBlock,16, 16 * wide_factor, num_blocks  , stride=1 )
        self.stage2 = self._make_layer(WideBasicBlock,16 * wide_factor, 32 * wide_factor, num_blocks , stride=2 )
        self.stage3 = self._make_layer(WideBasicBlock,32 * wide_factor, 64 * wide_factor, num_blocks , stride=2 )

        self.bn1 = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU()

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64 * wide_factor, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



    def _make_layer(self,block,in_channels,out_channels,num_block,stride):
        layers = []
        layers.append(block(in_channels,out_channels,stride))
        for i in range(1,num_block):
            layers.append(block(out_channels,out_channels,1))

        return nn.Sequential(*layers)

