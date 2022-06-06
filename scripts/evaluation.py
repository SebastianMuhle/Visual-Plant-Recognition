#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.

from typing import Tuple
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


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
        loss += loss_function(output, labels) * dataloader.batch_size

        output_max_scores, output_max_idx = output.max(dim=1)

        conf_matrix += confusion_matrix(labels, output_max_idx, labels=np.arange(num_classes))

        correct += (output_max_idx == labels).float().sum()

    accuracy = 100.0 * correct / (len(dataloader) * dataloader.batch_size)
    loss /= len(dataloader) * dataloader.batch_size
    return loss, accuracy, conf_matrix


def per_class_accuracy(conf_matrix: np.ndarray) -> Figure:
    # TODO: remove +1 when evaluating the real dataset
    num_correct = np.diag(conf_matrix) + 1
    num_total = np.sum(conf_matrix, axis=1) + 1

    accuracy = num_correct / num_total

    fig, ax = plt.subplots(1, 1)
    ax.hist(accuracy, 20, (0.0, 1.0), density=True)
    # plt.show()
    return fig


def worst_classes(conf_matrix: np.ndarray, k: int) -> Figure:
    # TODO: remove +1 when evaluating the real dataset
    num_correct = np.diag(conf_matrix) + 1
    num_total = np.sum(conf_matrix, axis=1) + 1

    accuracy = num_correct / num_total
    worst_k_idx = np.argsort(accuracy)[:k]

    worst_k = conf_matrix[worst_k_idx, :]

    fig, axs = plt.subplots(k, 1)
    for predictions, ax in zip(worst_k, axs):
        ax.bar(np.arange(predictions.size) + 1, predictions, align="center")
    # plt.show()
    return fig
