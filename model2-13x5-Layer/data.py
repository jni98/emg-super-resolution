"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Custom_Dataset, which is a custom dataset class for loading the data.

The code is called from every file that is loading a dataset.
"""
import scipy
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset
import torch
import numpy as np


class Custom_Dataset(Dataset):
    """
    Custom dataset class for loading the data

    """

    def __init__(self, data, directory, mean: float, std: float, spatial_whitening: str, bandpass_filter: bool) -> None:
        super().__init__()
        self.data = data
        self.directory = directory

        self.mean = mean
        self.std = std

        self.spatial_whitening = spatial_whitening
        self.bandpass_filter = bandpass_filter

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
        if self.bandpass_filter:
            lr_emg = self.butter_bandpass_filter(lr_emg, 20, 500, 2048, order=2)
        if self.spatial_whitening:
            lr_emg = self.spatial_whitening(lr_emg)

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
        if self.bandpass_filter:
            hr_emg = self.butter_bandpass_filter(hr_emg, 20, 500, 2048, order=2)
        if self.spatial_whitening:
            hr_emg = self.spatial_whitening(hr_emg)

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

    def spatial_whitening(self, data):
        """
        Apply spatial whitening to the data
        Parameters
        ----------
        data: np.array
            Data to be filtered

        Returns
        -------
        data: np.array
            Filtered data
        """
        # Reshape data for covariance matrix computation
        height, width, samples = data.shape
        reshaped_data = np.reshape(data, (height * width, samples))

        # Compute covariance matrix
        cov_matrix = np.cov(reshaped_data, rowvar=False)

        # Compute whitening transformation matrix
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        whitening_matrix = np.dot(np.dot(eig_vecs, np.diag(1.0 / np.sqrt(eig_vals + 1e-5))), eig_vecs.T)

        # Apply whitening transformation
        whitened_data = np.dot(reshaped_data, whitening_matrix).reshape((height, width, samples))

        return whitened_data

