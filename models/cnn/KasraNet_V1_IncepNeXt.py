"""
KasraNet-V1-IncepNeXt: Hybrid Multi-Scale Grouped Convolutional Neural Network.

This module implements `IncepNextblock` and `KasraNet_V1_IncepNeXt`, combining core
design principles from ResNeXt, Inception, and Pre-Activation WideResNet:

Architectural Highlights:
    - Shared Pre-Activation: Applies `BatchNorm2d -> ReLU` once on the block input
      before branching, reducing redundant memory reads and activations.
    - Parallel Multi-Scale Grouped Convolutions: Extracts spatial features across
      multiple receptive fields simultaneously using grouped 1x1, 3x3 (padding=1),
      and 5x5 (padding=2) convolutions (`groups=cardinality`).
    - Cross-Group Channel Mixing: Follows the summed multi-scale branches with a
      `BatchNorm2d -> ReLU -> 1x1 Conv (groups=1)` fusion layer to prevent group
      isolation across deep stages.
    - Identity & Projection Shortcuts: Preserves gradient flow via clean residual
      connections with 1x1 projection when spatial resolution or channel width changes.
    - Configurable Scaling: Supports dynamic depth (`num_layers = 6n + 4`),
      width scaling (`wide_factor`), and grouped capacity (`cardinality`).
      With `num_layers=28` and `wide_factor=2`, the model contains ~1.05M parameters.

--------------------------------------------------------------------------------

معماری ترکیبی شبکه عصبی کانولوشنی چندمقیاسه و گروهی (KasraNet-V1-IncepNeXt):

این ماژول کلاس‌های `IncepNextblock` و `KasraNet_V1_IncepNeXt` را پیاده‌سازی می‌کند
که ترکیبی از ایده‌های کلیدی ResNeXt، Inception و WideResNet (با Pre-Activation) است:

ویژگی‌های کلیدی معماری:
    - پیش‌فعال‌سازی مشترک (Shared Pre-Activation): اعمال یک‌باره `BatchNorm2d -> ReLU`
      روی ورودی بلوک قبل از انشعاب شاخه‌ها برای کاهش سربار حافظه و محاسبات تکراری.
    - کانولوشن‌های موازی چندمقیاسه و گروهی: استخراج هم‌زمان ویژگی‌های مکانی با میدان‌های دید
      مختلف از طریق شاخه‌های گروهی 1x1، 3x3 (padding=1) و 5x5 (padding=2).
    - ترکیب‌کننده بین‌کانالی (Channel Mixer): استفاده از لایه `BatchNorm2d -> ReLU -> 1x1 Conv`
      با `groups=1` پس از جمع شاخه‌ها برای ادغام اطلاعات بین تمام گروه‌ها و جلوگیری از ایزوله شدن آن‌ها.
    - اتصال‌های میان‌بُر (Shortcuts): حفظ جریان گرادیان با اتصال مستقیم (Identity) و یا
      نگاشت 1x1 هنگام تغییر ابعاد مکانی یا تعداد کانال‌ها.
    - مقیاس‌پذیری پویا: پشتیبانی از عمق دلخواه (`num_layers = 6n + 4`)، ضریب عرض (`wide_factor`)
      و تعداد گروه‌ها (`cardinality`). در حالت `num_layers=28` و `wide_factor=2` دارای حدود
      ۱.۰۵ میلیون پارامتر است.
"""


import torch
import torch.nn as nn

class IncepNextblock(nn.Module):
    def __init__(self,in_channel,out_channel,stride =1 ,cardinality = 16):
        super().__init__()
        assert in_channel % cardinality == 0, f"in_channel ({in_channel}) must be divisible by cardinality ({cardinality})"
        assert out_channel % cardinality == 0, f"out_channel ({out_channel}) must be divisible by cardinality ({cardinality})"


        self.preact = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU()
        )

        self.feature_extractor_1in1 = nn.Sequential(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size=1,
                stride=stride,
                groups = cardinality,
                bias= False
            )

        )

        self.feature_extractor_3in3 = nn.Sequential(
            nn.Conv2d(in_channel,
                      out_channel,
                      kernel_size=3,
                      padding=1,
                      stride=stride,
                      groups= cardinality,
                      bias= False
                      )

        )

        self.feature_extractor_5in5 = nn.Sequential(
            nn.Conv2d(in_channel,
                      out_channel,
                      kernel_size= 5,
                      padding= 2,
                      stride= stride,
                      groups= cardinality,
                      bias= False
                      )
        )

        self.feature_mixer = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(
                out_channel,
                out_channel,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=False
            )
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channel,
                    out_channel,
                    kernel_size=1,
                    stride = stride,
                    bias=False
                )
            )



    def forward(self,x):
        out = self.preact(x)
        multi_scale = (
            self.feature_extractor_1in1(out)
            + self.feature_extractor_3in3(out)
            + self.feature_extractor_5in5(out)
        )

        return self.feature_mixer(multi_scale) + self.shortcut(x)


class KasraNet_V1_IncepNeXt(nn.Module):
    def __init__(self,in_channels,num_class,num_layers,cardinality = 16,wide_factor = 2 ):
        super().__init__()
        num_block = (num_layers - 4) // 6
        c1 = 16 * wide_factor
        c2 = 32 * wide_factor
        c3 = 64 * wide_factor
        c4 = 128 * wide_factor


        self.conv1 = nn.Conv2d(in_channels,c1,kernel_size=3,padding=1,bias=False)
        self.stage1 = self._make_layer(IncepNextblock,c1,c2,1,cardinality,num_block= num_block)
        self.stage2 = self._make_layer(IncepNextblock,c2,c3,2,cardinality,num_block= num_block)
        self.stage3 = self._make_layer(IncepNextblock,c3,c4,2,cardinality,num_block= num_block)

        self.bn1 = nn.BatchNorm2d(c4)
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(c4,num_class)



    def _make_layer(self,block,in_channel,out_channel,stride,cardinality,num_block):

        layers = []
        layers.append(block(in_channel,out_channel,stride,cardinality))

        for _ in range(1,num_block):
            layers.append(block(out_channel, out_channel, 1, cardinality))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x