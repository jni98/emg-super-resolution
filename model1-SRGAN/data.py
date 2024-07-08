"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Custom_Dataset, which is a custom dataset class for loading the data.

The code is called from every file that is loading a dataset.
"""
from torch.utils.data import Dataset
import torch
import numpy as np
from emg_augmentations import GaussianNoise, WaveletDecomposition
import random
from utils import grid_interpolation, butter_bandpass_filter, standardization


class Custom_Dataset(Dataset):
    """
    Custom dataset class for loading the data

    """

    def __init__(self, data, directory, mean: float, std: float, augment: bool) -> None:
        """
        Constructs all the necessary attributes for the Custom_Dataset object.
        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe containing the data
        directory: str
            Directory where the data is stored
        mean: float
            Mean used for standardization
        std: float
            Standard deviation used for standardization
        augment: bool
            Boolean indicating whether to augment the data
        """
        super().__init__()
        self.data = data
        self.directory = directory

        self.mean = mean
        self.std = std
        self.augment = augment

        self.gaussian_noise_generator = GaussianNoise(target_snr__db=5.0)
        self.wavelet_decomposition = WaveletDecomposition(b=0.25, wavelet="db7", level=5, nr_of_grids=1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        """
        Returns the data at the given index
        Parameters
        ----------
        index: int

        Returns
        -------
        lr_emg_standardized: torch.tensor
            Low resolution EMG data standardized
        hr_emg_standardized: torch.tensor
            High resolution EMG data standardized
        lr_emg_path: str
            Path to the low resolution EMG data
        hr_emg_path: str
            Path to the high resolution EMG data
        """

        if torch.is_tensor(index):
            index = index.tolist()

        hr_emg_path = self.data.iloc[index]['hr_filename']
        hr_emg = np.load(self.directory + '/' + self.data.iloc[index, 0])

        if self.augment:
            # Randomly choose whether to augment the data or not
            augmentation_method = random.choice(
                [self.no_augmentation, self.apply_gaussian_noise, self.apply_wavelet_decomposition])

            # Apply the selected augmentation method
            hr_emg = augmentation_method(hr_emg)

        # Bandpass filter
        hr_emg = butter_bandpass_filter(hr_emg, 20, 500, 2048, order=2)

        # extract lr emg
        lr_emg = hr_emg[::2, ::2]

        # interpolate for hr shape
        lr_emg_interpolated = grid_interpolation(lr_emg, hr_emg.shape, method='linear')

        # Create lr file path
        lr_emg_path = hr_emg_path.replace("hr", "lr")

        # standardize both
        lr_emg_standardized = standardization(lr_emg, self.mean, self.std)
        hr_emg_standardized = standardization(hr_emg, self.mean, self.std)
        lr_emg_interpolated = standardization(lr_emg_interpolated, self.mean, self.std)

        # Change to torch and shape with channel = 1
        lr_emg_standardized_tensor = torch.tensor(lr_emg_standardized).unsqueeze(0).to(torch.float32)
        hr_emg_standardized_tensor = torch.tensor(hr_emg_standardized).unsqueeze(0).to(torch.float32)
        lr_emg_interpolated_tensor = torch.tensor(lr_emg_interpolated).unsqueeze(0).to(torch.float32)

        return lr_emg_standardized_tensor, hr_emg_standardized_tensor, lr_emg_path, hr_emg_path, lr_emg_interpolated_tensor

    def no_augmentation(self, data):
        return data

    def apply_gaussian_noise(self, data):
        return self.gaussian_noise_generator(data)

    def apply_wavelet_decomposition(self, data):
        return self.wavelet_decomposition(data)