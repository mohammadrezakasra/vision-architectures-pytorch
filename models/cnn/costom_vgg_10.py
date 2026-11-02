"""
معماری VGG10 سفارشی (ترکیبی از فلسفه هسته‌های بزرگ AlexNet و ساختار بلوکی VGG)

هدف آزمایشی (Experimental Motivation):
--------------------------------------
این معماری برای ارزیابی تجربی فرضیه کلیدی مقاله VGG (Simonyan & Zisserman) طراحی شده است:
«آیا استفاده از تعداد لایه‌های عمیق‌تر با هسته‌های کوچک ۳×۳ بر لایه‌های کمتر با هسته‌های بزرگ برتری دارد؟»

نکات و مقایسه معماری با VGG16 و AlexNet:
-----------------------------------------
۱. هسته‌های بزرگ در برابر پشته هسته‌های ۳×۳ (Kernel Size vs Depth):
   - در VGG16 تمام لایه‌ها منحصراً از هسته‌های کوچک ۳×۳ استفاده می‌کنند؛ چون دو لایه ۳×۳ میدان دیدی
     معادل ۵×۵ و پنج لایه ۳×۳ میدان دیدی معادل ۱۱×۱۱ ایجاد می‌کنند، در حالی که پارامتر کمتری مصرف
     کرده و توابع فعال‌ساز (ReLU) بیشتری برای استخراج الگوهای غیرخطی فراهم می‌آورند.
   - در این مدل سفارشی (VGG10)، مشابه رویکرد AlexNet، از هسته‌های اولیه بسیار بزرگ (۱۱×۱۱، ۷×۷ و ۵×۵)
     با عمق کمتر استفاده شده است تا سرعت رشد میدان دید (Receptive Field) در لایه‌های ابتدایی تست شود
     و رفتار مدل از نظر پایداری گرادیان و قدرت تعمیم با VGG16 مقایسه گردد.

۲. ساختار هیبریدی AlexNet-VGG:
   - لایه‌های اولیه به دلیل اندازه هسته‌ها (۱۱ و ۷ و ۵) رفتاری بسیار نزدیک به AlexNet دارند، اما
     پدینگ‌ها به گونه‌ای تنظیم شده‌اند که ابعاد تصویر حفظ شود (Exact Same Padding) و کاهش ابعاد فضایی
     صرفاً توسط ۵ مرحله MaxPool2d انجام گیرد.
   - لایه‌های انتهایی همچنان ساختار کانولوشن‌های عمیق ۳×۳ مدل VGG را حفظ کرده‌اند.

۳. وجود Batch Normalization:
   - مشابه نسخه مدرن پیاده‌سازی‌شده، به لایه‌ها BatchNorm اضافه شده تا آموزش هسته‌های سنگین ۱۱×۱۱ و ۷×۷
     با پایداری کامل و بدون انفجار/میرایی گرادیان انجام پذیرد.

۴. طبقه‌بند سبک (Light Classifier):
   - برای دیتاست‌های با رزولوشن محدود (نظیر CIFAR-10)، طبقه‌بند قیفی سبک جایگزین لایه‌های سنگین ۴۰۹۶
     شده تا بیش‌برازش به حداقل برسد.
"""

import torch
import torch.nn as nn

class VGG10(nn.Module):
    def __init__(self,input_size = 224 ,input_channels=3 , num_classes=1000 ):
        super().__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels,128,kernel_size=11,stride=1,padding=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128,256,kernel_size=7,stride=1,padding=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

        )

        inputs = self.input_size // 32

        # -------------------------------------------------------------
        # طبقه‌بند اصلی (ImageNet / مناسب ابعاد ۲۲۴)
        # -------------------------------------------------------------
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * inputs * inputs, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(4096, self.num_classes)
        # )

        # -------------------------------------------------------------
        # طبقه‌بند سبک (Light Classifier - بهینه‌شده برای جلوگیری از Overfitting)
        # -------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(512 * inputs * inputs,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256,self.num_classes)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1,-1)
        x = self.classifier(x)
        return x
