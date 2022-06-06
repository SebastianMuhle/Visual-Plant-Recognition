#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.


def train_epoch(model, dataloader, optimizer, loss_function, use_cuda):
    model.train()
    for batch_idx, data in enumerate(dataloader):
        print(batch_idx)
        if batch_idx > 5:
            break
        images = data["image"]
        labels = data["plant_label"]
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(images)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
