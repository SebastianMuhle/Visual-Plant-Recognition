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


class Oxford102Dataset(Dataset):
    def __init__(self, csv_file: str, dataset_dir = str, transform = None):
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

