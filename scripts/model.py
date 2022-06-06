#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# define the CNN architecture
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        # Max pooling layer (divides image size by 2)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 28 * 28, 500)
        self.fc2 = nn.Linear(500, 102)

        # softmax
        self.sm = nn.Softmax(dim=1)

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten image input
        x = x.view(-1, 128 * 28 * 28)
        # Dropout layer
        x = self.dropout(x)
        # 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # Dropout layer
        x = self.dropout(x)
        # 2nd hidden layer
        x = self.fc2(x)
        return self.sm(x)


def resNet_152(number_of_features: int, child_layers_to_freeze: int):
    model = models.resnet152(pretrained=True)
    for ct, child in enumerate(model.children()):
        print(" child", ct, "is -")
        print(child)
        if ct <= child_layers_to_freeze:
            for param in child.parameters():
                param.requires_grad = False
    model.fc = nn.Linear(number_of_features, 102)

    return model
