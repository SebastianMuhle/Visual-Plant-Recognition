#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.
import torch


def train_epoch(model, dataloader, optimizer, loss_function, use_cuda, use_mps):
    model.train()
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
        optimizer.zero_grad()
        output = model(images)
        # TODO: set reduction to "mean"
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
