#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.

def train_epoch(model, dataloader, optimizer, loss_function, use_cuda, progress_bar):
    model.train()
    for batch_idx, data in enumerate(dataloader):
        images = data['image']
        labels = data['plant_label']
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(images)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

        progress_bar['batch'](batch_idx + 1)

        progress_bar['progress'].update(step=1)

    # # print training/validation statistics
    # print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, loss))
