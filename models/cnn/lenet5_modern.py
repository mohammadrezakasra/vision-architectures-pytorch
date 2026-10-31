import torch
import torch.nn as nn

class LeNet5_M(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2 ,stride=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5,stride=2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.classifier(x)
        return x