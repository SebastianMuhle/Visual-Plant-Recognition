#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.

import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix


def evaluation(
    model: nn.Module, dataloader: DataLoader, loss_function: nn.CrossEntropyLoss, use_cuda: bool
) -> tuple[float, float, np.ndarray]:
    loss = 0
    correct = 0
    num_classes = 102
    conf_matrix = np.zeros((num_classes, num_classes))

    for batch_idx, data in enumerate(dataloader):
        images = data["image"]
        labels = data["plant_label"]
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(images)
        loss += loss_function(output, labels).detach().item() * dataloader.batch_size

        output_max_scores, output_max_idx = output.max(dim=1)

        conf_matrix += confusion_matrix(labels, output_max_idx, labels=np.arange(num_classes))

        correct += (output_max_idx == labels).float().sum()

    accuracy = 100.0 * correct / (len(dataloader) * dataloader.batch_size)
    loss /= len(dataloader) * dataloader.batch_size
    return loss, accuracy, conf_matrix
