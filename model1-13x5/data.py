"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Custom_Dataset, which is a custom dataset class for loading the data.

The code is called from every file that is loading a dataset.
"""

from torch.utils.data import Dataset
import torch
import numpy as np
from scipy.signal import butter, filtfilt


class Custom_Dataset(Dataset):
    """
    Custom dataset class for loading the data

    """
    def __init__(self, data, mean: float, std: float) -> None:
        super().__init__()
        self.data = data

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

        lr_emg = np.load(self.data.iloc[index, 0])
        lr_emg = self.reshape_3D_to_2D(lr_emg)

        # bandpass filter the signal
        lr_emg = self.butter_bandpass_filter(lr_emg, 20, 500, 2048)
        # Standardize using mean and std values of the complete dataset
        lr_emg_standardized = self.standardization(lr_emg, self.mean, self.std)
        # unsqueeze to add a one channel dimension at the beginning
        lr_emg_standardized = torch.tensor(lr_emg_standardized).unsqueeze(0).to(torch.float32)

        hr_emg = np.load(self.data.iloc[index, 1])
        hr_emg = self.reshape_3D_to_2D(hr_emg)

        # bandpass filter the signal
        hr_emg = self.butter_bandpass_filter(hr_emg, 20, 500, 2048)
        #  Standardize using mean and std values of the complete dataset
        hr_emg_standardized = self.standardization(hr_emg,self.mean,self.std)
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
        standardized_data = (data - mean)/std
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
        print("Signal filtered")
        return filtered_signal

    def reshape_3D_to_2D(self, data: np.array) -> np.array:
        """
        Reshape the 3D data to 2D
        Parameters
        ----------
        data: np.array
            3D data

        Returns
        -------
        reshaped_data: np.array
            2D data
        """
        data_shape = data.shape
        data[:, 1::2, :] = np.flipud(data[:, 1::2, :])
        reshaped_data = np.flipud(data).transpose(1, 0, 2).reshape(65, data_shape[2])
        return reshaped_data