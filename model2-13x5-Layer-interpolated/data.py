"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Custom_Dataset, which is a custom dataset class for loading the data.

The code is called from every file that is loading a dataset.
"""
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset
import torch
import numpy as np
from emg_augmentations import GaussianNoise


class Custom_Dataset(Dataset):
    """
    Custom dataset class for loading the data

    """

    def __init__(self, data, directory, mean: float, std: float, gaussian_noise: bool) -> None:
        super().__init__()
        self.data = data
        self.directory = directory

        self.mean = mean
        self.std = std
        self.gaussian_noise = gaussian_noise

        # Instantiate the augmentation class with optional parameters
        self.gaussian_noise_augmentor = GaussianNoise(target_snr__db=5.0)  # Adjust parameters as needed

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

        # load file paths
        lr_emg_path = self.data.iloc[index]['lr_filename']
        hr_emg_path = self.data.iloc[index]['hr_filename']

        # load and preprocess lr data
        lr_emg = np.load(self.directory + '/' + self.data.iloc[index, 0])

        # if self.gaussian_noise:
        #     lr_emg = self.gaussian_noise_augmentor(lr_emg)
        #     print("augmentation done")

        # bandpass filter
        lr_emg = self.butter_bandpass_filter(lr_emg, 20, 500, 2048, order=2)

        # Standardize using mean and std values of the complete dataset
        lr_emg_standardized = self.standardization(lr_emg, self.mean, self.std)

        # unsqueeze to add a one channel dimension at the beginning
        lr_emg_standardized_copy = lr_emg_standardized.copy()
        lr_emg_standardized_tensor = torch.tensor(lr_emg_standardized_copy).unsqueeze(0).to(torch.float32)

        # load and preprocess hr data
        hr_emg = np.load(self.directory + '/' + self.data.iloc[index, 1])

        # if self.gaussian_noise:
        #     hr_emg = self.gaussian_noise_augmentor(hr_emg)

        # bandpass filter
        hr_emg = self.butter_bandpass_filter(hr_emg, 20, 500, 2048, order=2)

        #  Standardize using mean and std values of the complete dataset
        hr_emg_standardized = self.standardization(hr_emg, self.mean, self.std)

        # unsqueeze to add a one channel dimension at the beginning
        hr_emg_standardized_copy = hr_emg_standardized.copy()
        hr_emg_standardized_tensor = torch.tensor(hr_emg_standardized_copy).unsqueeze(0).to(torch.float32)

        return lr_emg_standardized_tensor, hr_emg_standardized_tensor, lr_emg_path, hr_emg_path

    def standardization(self, data: np.array, mean: float, std: float) -> np.array:
        """
        Standardize the data using the mean and std values
        Parameters
        ----------
        data: np.array
            Data to be standardized
        mean: float
            Mean value of the complete dataset
        std: float
            Standard deviation value of the complete dataset

        Returns
        -------
        standardized_data: np.array
            Standardized data
        """

        # Min-max normalization using precalculated min and max values
        standardized_data = (data - mean) / std
        return standardized_data

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        """
        Apply a bandpass filter to the data
        Parameters
        ----------
        data: np.array
            Data to be filtered
        lowcut: float
            Lower cut-off frequency
        highcut: float
            Higher cut-off frequency
        fs: int
            Sampling frequency
        order: int
            Order of the filter

        Returns
        -------
        y: np.array
            Filtered data
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, data)

        return filtered_signal