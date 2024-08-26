"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Custom_Dataset, which is a custom dataset class for loading the data.

The code is called from every file that is loading a dataset.
"""
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset
import torch
import numpy as np


class Custom_Dataset(Dataset):
    """
    Custom dataset class for loading the data

    """

    def __init__(self, data, directory, mean: float, std: float) -> None:
        super().__init__()
        self.data = data
        self.directory = directory

        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        if torch.is_tensor(index):
            index = index.tolist()

        # Load file paths
        hr_emg_path = self.data.iloc[index]['hr_filename']
        lr_emg_path = hr_emg_path.replace("hr", "lr")

        # Load and preprocess HR data
        hr_emg = np.load(f"{self.directory}/{hr_emg_path}")

        # extract lr emg
        lr_emg = hr_emg[::2, ::2]

        return (lr_emg, hr_emg,lr_emg_path, hr_emg_path)