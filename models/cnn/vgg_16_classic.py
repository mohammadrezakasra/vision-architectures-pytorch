"""
معماری VGG16 (نسخه بهینه‌شده برای رزولوشن‌های متغیر و CIFAR-10)

نکات فنی درباره تفاوت با مقاله مرجع (Simonyan & Zisserman, 2014):
--------------------------------------------------------------
۱. لایه‌های Batch Normalization (BatchNorm2d):
   نسخه اصلی مقاله فاقد BatchNorm بود (این لایه در سال ۲۰۱۵ معرفی شد). در این پیاده‌سازی
   مشابه نسخه VGG16_BN تورچ‌ویژن، برای پایدارسازی آموزش، تسریع همگرایی گرادیان و کاهش
   حساسیت به مقدار اولیه وزن‌ها، پس از هر کانولوشن از BatchNorm2d استفاده شده است.

۲. ساختار لایه‌های تمام‌متصل (Classifier):
   - در نسخه اصلی VGG-16، بخش طبقه‌بند از دو لایه مخفی ۴۰۹۶ تایی (با ابعاد 512*7*7 -> 4096 -> 4096 -> num_classes)
     تشکیل شده که بیش از ۱۰۰ میلیون پارامتر به مدل تحمیل می‌کند.
   - برای دیتاست‌های سبک یا کوچک با رزولوشن پایین (مثل CIFAR-10)، لایه‌های ۴۰۹۶ تایی به سرعت دچار بیش‌برازش (Overfitting)
     می‌شوند؛ به همین دلیل به صورت پیش‌فرض از نسخه سبک (Light Classifier با ساختار قیفی) استفاده شده است.
   - در صورت نیاز به استفاده از طبقه‌بند اصلی ImageNet، بلوک کامنت‌شده orginal Vgg16 را از حالت کامنت خارج کنید.
"""

import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self,input_size = 224 ,input_channels=3 , num_classes=1000 ):
        super().__init__()

        self.input_size = input_size
        self.input_channels = input_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
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
        # 1. Original VGG16 Classifier (برای ImageNet / دیتای بسیار حجیم)
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
        # 2. Light VGG16 Classifier (بهینه‌شده برای CIFAR-10 جهت جلوگیری از Overfit)
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
