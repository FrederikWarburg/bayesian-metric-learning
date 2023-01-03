import torch
import torch.nn as nn
from models.layers.normalization import L2Norm
import torch.nn.functional as F

"""
class MnistLinearNet(nn.Module):
    def __init__(self, latent_dim=2):
        super(MnistLinearNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim),
            L2Norm(),
        )

    def forward(self, x):

        x = self.cnn(x)
        x = self.linear(x)

        return {"z_mu": x}
"""


class Cifar10Net(nn.Module):
    def __init__(self, latent_dim=128, dropout_rate=0.0):
        super(Cifar10Net, self).__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            # nn.Dropout2d(0.25),
            nn.Flatten(),
        )

        if dropout_rate > 0:
            
            print("Using dropout rate: {}".format(dropout_rate))
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(32, 64, 3, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                nn.Dropout2d(dropout_rate),
                nn.Flatten(),
            )

        self.linear = nn.Sequential(nn.Linear(1024, latent_dim), L2Norm())

    def forward(self, x):
        
        x = self.backbone(x)
        x = self.linear(x)

        return {"z_mu": x}
