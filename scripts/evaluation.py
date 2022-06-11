#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix


def evaluation(
    model: nn.Module, dataloader: DataLoader, loss_function: nn.CrossEntropyLoss, use_cuda: bool, use_mps: bool
) -> tuple[float, float, np.ndarray]:
    with torch.no_grad():
        loss = 0
        correct = 0
        num_classes = 102
        conf_matrix = np.zeros((num_classes, num_classes))

        for batch_idx, data in enumerate(dataloader):
            images = data["image"]
            labels = data["plant_label"]
            # move to GPU
            # if use_cuda:
            # data, target = data.cuda(), target.cuda()
            if use_mps:
                device = torch.device("mps")
                if device:
                    images.to(device)
                    labels.to(device)
            output = model(images)
            # TODO: set reduction to sum
            loss += loss_function(output, labels).item() * dataloader.batch_size

            output_max_scores, output_max_idx = output.max(dim=1)

            conf_matrix += confusion_matrix(labels, output_max_idx, labels=np.arange(num_classes))
            correct += (output_max_idx == labels).float().sum()

        accuracy = 100.0 * correct / (len(dataloader) * dataloader.batch_size)
        loss /= len(dataloader) * dataloader.batch_size
        return loss, accuracy, conf_matrix
