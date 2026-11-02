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

        # orginal Vgg16
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * inputs * inputs,4096),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(4096,4096),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(4096,self.num_classes)
        # )

        #light Vgg16
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
