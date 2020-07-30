import torch
import numpy as np
from torch import nn, optim

class Alexnet(nn.Module):
    def __init__(self, num_classes):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            #conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            #conv2
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            #conv3
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            #conv4
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),

            #conv5
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            #fc6
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            #fc7
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),

            #fc8
            nn.Linear(4096, num_classes)
        )
    def forward(self, input):
        x = self.features(input)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x