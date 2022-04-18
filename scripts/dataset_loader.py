#
# BSD 3-Clause License
#
#
# Copyright (c) 2022, Sebastian Muhle, Dominik Muhle.
# All rights reserved.

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import numpy as np


class Oxford102Dataset(Dataset):
    def __init__(self, csv_file: str, dataset_dir=str, transform=None):
        """
        Args:
        :param csv_file (string): Path to the csv file with the image labels
        :param dataset_dir (string): Directory with all the images
        :param transform (callable, optional): Optional transforms to be applied on the image
        """
        self.labels = pd.read_csv(csv_file)
        self.dataset = dataset_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        plant_number = idx + 1
        img_name = self.dataset + "jpg/image_" + str(plant_number).zfill(5) + ".jpg"
        image = io.imread(img_name)
        plant_label = self.labels["labels"][idx]

        sample = {'image': image, 'plant_label': plant_label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """
    Rescales the image in a sample to a given size
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, plant_label = sample["image"], sample["plant_label"]

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'plant_label': plant_label}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, plant_label = sample['image'], sample['plant_label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        plant_label = plant_label

        return {'image': image, 'plant_label': plant_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, plant_label = sample['image'], sample['plant_label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(plant_label)}


