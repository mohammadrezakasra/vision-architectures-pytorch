"""
===============================================================================
Dual-Branch Residual Network (DualBranchResNet-28)
معماری رزنت دوشاخه‌ای ۲۸ لایه (DualBranchResNet-28)
===============================================================================

[EN] Overview:
This module provides a standalone, from-scratch PyTorch implementation of a 28-layer
Dual-Branch Residual Network designed for CIFAR-10 benchmarking. It combines the
stability of local residual shortcuts with multi-branch feature extraction:

1. Block-Level Dual-Branch Design (Split-Transform-Merge):
   Instead of widening a single sequential path (as in WideResNet), each residual
   block splits the input tensor into two independent, parallel convolutional
   branches (`feature_extractor_1` and `feature_extractor_2`). Both branches
   process the input concurrently and merge via element-wise addition with the
   local skip connection:
   Block Output = Shortcut(x) + Branch_1(x) + Branch_2(x)
   (Architecturally equivalent to a Pre-Activation ResNeXt with Cardinality = 2).

2. Local Pre-Activation Residual Shortcuts:
   Keeping skip connections local to every 2-layer block preserves short gradient
   highways and prevents training instability caused by long macro-shortcuts.
   Each block uses the Pre-Activation layout (BatchNorm -> ReLU -> Conv2d) with
   1x1 projection shortcuts when spatial resolution or channel dimensions change.

Purpose in Repository:
Built to compare block-level dual-branch parallelism against single-branch
WideResNet-28 and higher-cardinality ResNeXt-29 on CIFAR-10.

-------------------------------------------------------------------------------

[FA] معرفی معماری:
این فایل شامل پیاده‌سازی کامل و مستقل (از صفر) یک شبکه رزنت دوشاخه‌ای ۲۸ لایه در
فریم‌ورک PyTorch است که برای بنچمارک روی دیتاست CIFAR-10 طراحی شده است:

۱. طراحی دوشاخه‌ای در سطح بلوک (Block-Level Dual-Branch):
   برخلاف WideResNet که عرض یک مسیر واحد را افزایش می‌دهد، در این معماری درون هر
   بلوک رزنت دو مسیر کانولوشنی موازی و مستقل (`feature_extractor_1` و
   `feature_extractor_2`) قرار دارد که ورودی را به صورت هم‌زمان پردازش کرده و در
   انتهای همان بلوک با میان‌بر محلی جمع می‌شوند:
   خروجی بلوک = میان‌بر(x) + شاخه اول(x) + شاخه دوم(x)
   (این ساختار از نظر ریاضی معادل ResNeXt با تعداد شاخه C=2 روی بلوک‌های پایه است).

۲. حفظ پایداری با میان‌برهای محلی و پیش‌فعال‌سازی (Pre-Activation):
   محلی ماندن شورت‌کات‌ها در سطح هر بلوک، مسیر کوتاه جریان گرادیان را حفظ کرده و
   از ناپایداری ناشی از شورت‌کات‌های طولانی جلوگیری می‌کند. تمامی لایه‌ها با ترتیب
   (BatchNorm -> ReLU -> Conv2d) پیاده‌سازی شده‌اند.

هدف در مخزن:
مقایسه عملکرد، پایداری آموزش و تعداد پارامترهای موازی‌سازی دوشاخه‌ای در سطح بلوک
با معماری‌های WideResNet-28 و ResNeXt-29 روی دیتاست CIFAR-10.
===============================================================================
"""


import torch
import torch.nn as nn


class DualBranchBaseBlock(nn.Module):
    def __init__(self,in_channels ,out_channels ,stride =1):
        super().__init__()

        self.feature_extractor_1 = nn.Sequential(

            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,out_channels,kernel_size= 3,padding= 1,stride= stride,bias=False),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,bias= False)

        )
        self.feature_extractor_2 = nn.Sequential(

            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)

        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size= 1,stride=stride , bias= False)
            )

    def forward(self,x):
        return self.shortcut(x) + self.feature_extractor_1(x) + self.feature_extractor_2(x)



class DualBranchResNet28(nn.Module):
    def __init__(self,in_channels = 3,depth = 28,num_classes = 10):
        super().__init__()
        num_blocks = (depth - 4) // 6

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1,padding=1,bias= False)

        self.stage1 = self._make_layer(DualBranchBaseBlock,16,16,stride = 1 , num_block= num_blocks)
        self.stage2 = self._make_layer(DualBranchBaseBlock,16,32,stride = 2 , num_block= num_blocks)
        self.stage3 = self._make_layer(DualBranchBaseBlock,32,64,stride = 2 , num_block= num_blocks)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc =  nn.Linear(64,num_classes)

    def _make_layer(self,block,in_channels,out_channels,stride,num_block):
        layers = []
        layers.append(block(in_channels,out_channels,stride))
        for _ in range(1,num_block):
            layers.append(block(out_channels,out_channels,stride=1))

        return nn.Sequential(*layers)



    def forward(self,x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1) # برای بچ سایز
        x = self.fc(x)
        return x


