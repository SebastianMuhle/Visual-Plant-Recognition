#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.

from typing import Tuple
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluation(model: nn.Module, dataloader: DataLoader, loss_function: nn.CrossEntropyLoss, use_cuda: bool) -> tuple[float, float]:
    model.eval()
    loss = 0
    correct = 0
    for batch_idx, data in enumerate(dataloader):
        images = data['image']
        labels = data['plant_label']
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        output = model(images)
        loss += loss_function(output, labels) * dataloader.batch_size

        output_max_scores, output_max_idx = output.max(dim=1)

        correct += (output_max_idx == labels).float().sum()

    accuracy = 100.0 * correct / (len(dataloader) * dataloader.batch_size)
    loss /= (len(dataloader) * dataloader.batch_size)
    return loss, accuracy
