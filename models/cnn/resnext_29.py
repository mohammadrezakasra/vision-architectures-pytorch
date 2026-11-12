"""
===============================================================================
Pre-Activation ResNeXt-29 (Aggregated Residual Transformations)
معماری ResNeXt-29 با ساختار پیش‌فعال‌سازی (Pre-Activation)
===============================================================================

[EN] Overview:
This module provides a standalone, from-scratch PyTorch implementation of the
ResNeXt-29 architecture designed for CIFAR-10 benchmarking. It introduces
"Cardinality" (the size of the set of transformations) as a third dimension
alongside depth and width:

1. Split-Transform-Merge via Grouped Convolutions:
   Instead of widening a single path or manually instantiating separate parallel
   branches, each Bottleneck block splits channels into `C` independent groups
   using `nn.Conv2d(..., groups=cardinality)`. This achieves multi-branch
   representational power with high GPU efficiency and fewer parameters.

2. Design Choice — Why Pre-Activation Instead of Standard Post-Activation?
   While the original 2017 ResNeXt paper uses Post-Activation (Conv -> BN -> ReLU),
   this implementation deliberately adopts the Pre-Activation layout
   (BatchNorm -> ReLU -> Conv2d) for two key engineering reasons:
   - Fair Benchmarking (Apples-to-Apples Comparison): Both `WideResNet-28` and
     `DualBranchResNet-28` in this repository use Pre-Activation blocks. Keeping
     the activation/normalization layout identical isolates the true architectural
     effect of Cardinality (Grouped Convolutions) during CIFAR-10 evaluation.
   - Unobstructed Gradient Flow: Pre-Activation keeps the residual skip connection
     as a clean identity highway, improving regularization and training stability.

Purpose in Repository:
Serves as the grouped-convolution baseline (e.g., ResNeXt-29, 8x64d) to benchmark
against WideResNet-28 and DualBranchResNet-28 under identical training setups.

-------------------------------------------------------------------------------

[FA] معرفی معماری:
این فایل شامل پیاده‌سازی مستقل و از صفر معماری ResNeXt-29 در فریم‌ورک PyTorch
برای ارزیابی روی دیتاست CIFAR-10 است. این معماری علاوه بر عمق (Depth) و عرض
(Width)، بُعد سومی به نام «کاردینالیتی» (Cardinality یا تعداد مسیرهای موازی) را
معرفی می‌کند:

۱. الگوی Split-Transform-Merge با کانولوشن گروهی (Grouped Convolution):
   به جای تعریف دستی چندین شاخه مجزا، در لایه میانی (۳×۳) هر بلوک Bottleneck با
   استفاده از پارامتر `groups=cardinality`، کانال‌ها به C گروه کاملاً مستقل و
   موازی تقسیم می‌شوند که هم سرعت اجرای بالایی روی GPU دارد و هم پارامترها را بهینه
   نگه می‌دارد.

۲. تصمیم طراحی — چرا از Pre-Activation به جای Post-Activation استاندارد استفاده شد؟
   اگرچه در مقاله اصلی ResNeXt (سال ۲۰۱۷) از ترتیب استاندارد (Conv -> BN -> ReLU)
   استفاده شده است، اما در این پیاده‌سازی به دو دلیل فنی مهم از ساختار
   پیش‌فعال‌سازی (BatchNorm -> ReLU -> Conv2d) استفاده شده است:
   - مقایسه عادلانه در بنچمارک (Fair Benchmarking): از آنجا که مدل‌های
     `WideResNet-28` و `DualBranchResNet-28` در این مخزن با ساختار Pre-Activation
     پیاده‌سازی شده‌اند، یکسان‌سازی ترتیب لایه‌ها باعث می‌شود در آزمایش‌های CIFAR-10،
     تأثیر خالصِ «کاردینالیتی و کانولوشن گروهی» بدون خطای ناشی از تفاوت محل BatchNorm
     سنجیده شود.
   - جریان آزاد گرادیان (Unobstructed Gradient Flow): در حالت Pre-Activation، مسیر
     میان‌بر (Shortcut) کاملاً بدون مانع باقی می‌ماند که همگرایی و تعمیم‌پذیری مدل
     را روی تصاویر کوچک CIFAR-10 بهبود می‌بخشد.

هدف در مخزن:
ارزیابی عملکرد و بهره‌وری پارامتریِ کانولوشن‌های گروهی (مانند پیکربندی 8x64d) در
مقایسه مستقیم با WideResNet-28 و DualBranchResNet-28 تحت شرایط آموزش یکسان.
===============================================================================
"""


import torch
import torch.nn as nn

class ResNextBaseBlock(nn.Module):
    def __init__(self, in_channels,out_channels ,bottleneck_channels,stride,cardinality = 2):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels,bottleneck_channels,kernel_size = 1,stride=1, bias= False),

            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels,bottleneck_channels,kernel_size = 3,stride = stride,padding=1 ,groups= cardinality, bias= False),

            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels,out_channels,kernel_size = 1,stride=1, bias= False)

        )


        self.shortcut = nn.Sequential()
        if stride !=1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size =1,stride=stride , bias= False),
            )
    def forward(self, x):
        return self.shortcut(x) + self.feature_extractor(x)


class ResNext29(nn.Module):
    def __init__(self,in_channels,num_classes,depth):
        super().__init__()
        num_block = (depth - 2) //9




    def _make_layer(self,block,in_channel,out_channels,bottleneck_channels,stride,cardinality,num_block):
        layers = []
        layers.append(block(in_channel,out_channels,bottleneck_channels,stride,cardinality ))
        for i in range(1,num_block):
            layers.append(block(out_channels,out_channels,bottleneck_channels,stride = 1,cardinality = cardinality))

        return nn.Sequential(*layers)

    def forward(self):
        ...