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

        lr_emg_path = self.data.iloc[index]['lr_filename']
        hr_emg_path = self.data.iloc[index]['hr_filename']

        lr_emg = np.load(self.directory + '/' + self.data.iloc[index, 0])

        # bandpass filter
        lr_emg = self.butter_bandpass_filter(lr_emg, 20, 500, 2048, order=5)

        lr_emg = self.reshape(lr_emg)

        # plt.figure()
        # plt.plot(lr_emg[0][0])
        # plt.savefig('lr_emg.png')

        # Standardize using mean and std values of the complete dataset
        lr_emg_standardized = self.standardization(lr_emg, self.mean, self.std)

        # plt.figure()
        # plt.plot(lr_emg_standardized[0][0])
        # plt.savefig('lr_emg_standardized.png')

        # unsqueeze to add a one channel dimension at the beginning
        lr_emg_standardized = torch.tensor(lr_emg_standardized).unsqueeze(0).to(torch.float32)

        hr_emg = np.load(self.directory + '/' + self.data.iloc[index, 1])

        # bandpass filter
        hr_emg = self.butter_bandpass_filter(hr_emg, 20, 500, 2048, order=5)

        hr_emg = self.reshape(hr_emg)

        #  Standardize using mean and std values of the complete dataset
        hr_emg_standardized = self.standardization(hr_emg, self.mean, self.std )
        # unsqueeze to add a one channel dimension at the beginning
        hr_emg_standardized = torch.tensor(hr_emg_standardized).unsqueeze(0).to(torch.float32)

        return lr_emg_standardized, hr_emg_standardized, lr_emg_path, hr_emg_path

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

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
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

    def reshape(self, data: np.array) -> np.array:
        """
        Reshape the data to the correct shape
        Parameters
        ----------
        data: np.array
            Data to be reshaped

        Returns
        -------
        reshaped_data: np.array
            Reshaped data
        """
        # 3D to 2D
        data[:, 1::2, :] = np.flipud(data[:, 1::2, :])
        reshaped_data = np.flipud(data).transpose(1, 0, 2).reshape(data.shape[0]*data.shape[1], data.shape[2])

        # Permute to have samples x channel
        reshaped_data = reshaped_data.transpose()

        return reshaped_data
