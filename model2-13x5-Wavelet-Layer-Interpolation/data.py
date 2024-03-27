"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Custom_Dataset, which is a custom dataset class for loading the data.

The code is called from every file that is loading a dataset.
"""
import scipy
from matplotlib import pyplot as plt
from scipy.interpolate import RegularGridInterpolator
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

        # TODO: Checken, dass Eckpunkte übernommen werden
        # Perform linear interpolation
        interpolated_lr = self.grid_interpolation(lr_emg, (13, 5, lr_emg.shape[2]), 'linear')

        # Standardize using mean and std values of the complete dataset
        lr_emg_standardized = self.standardization(interpolated_lr, self.mean, self.std)

        # bandpass filter
        if self.bandpass_filter:
            lr_emg_standardized = self.butter_bandpass_filter(lr_emg_standardized, 20, 500, 2048, order=2)
        if self.spatial_whitening:
            lr_emg_standardized = self.spatial_whitening_function(lr_emg_standardized)

        # plt.figure()
        # plt.plot(lr_emg[0][0])
        # plt.savefig('lr_emg.png')

        # plt.figure()
        # plt.plot(lr_emg_standardized[0][0])
        # plt.savefig('lr_emg_standardized.png')

        # unsqueeze to add a one channel dimension at the beginning
        lr_emg_standardized_copy = lr_emg_standardized.copy()
        lr_emg_standardized_tensor = torch.tensor(lr_emg_standardized_copy).unsqueeze(0).to(torch.float32)

        hr_emg = np.load(self.directory + '/' + self.data.iloc[index, 1])

        #  Standardize using mean and std values of the complete dataset
        hr_emg_standardized = self.standardization(hr_emg, self.mean, self.std)

        # bandpass filter
        if self.bandpass_filter:
            hr_emg_standardized = self.butter_bandpass_filter(hr_emg_standardized, 20, 500, 2048, order=2)
        if self.spatial_whitening:
            hr_emg_standardized = self.spatial_whitening_function(hr_emg_standardized)


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

    def spatial_whitening_function(self, data):
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

    def grid_interpolation(self, lr_emg: np.ndarray, hr_emg_shape: tuple, method: str) -> np.ndarray:
        """
        Performs cubic interpolation on the low resolution data to get high resolution data.
        Parameters
        ----------
        lr_emg: numpy.ndarray
                Low resolution emg data.
        hr_emg_shape: tuple
                Shape of the high resolution emg data.
        method: str
                Method of interpolation. On default set to linear.

        Returns
        -------
        interpolated_signal: numpy.ndarray
                High resolution emg data.
        """
        # Create a grid for the original signal
        x = np.linspace(0, 1, lr_emg.shape[0])
        y = np.linspace(0, 1, lr_emg.shape[1])
        z = np.linspace(0, 1, lr_emg.shape[2])

        # Create a new grid for the desired interpolated signal
        new_x = np.linspace(0, 1, hr_emg_shape[0])
        new_y = np.linspace(0, 1, hr_emg_shape[1])
        new_z = np.linspace(0, 1, hr_emg_shape[2])

        # Create meshgrid from new grid
        mesh_x, mesh_y, mesh_z = np.meshgrid(new_x, new_y, new_z, indexing='ij')

        # Create interpolator
        interpolator = RegularGridInterpolator((x, y, z), lr_emg, method=method)

        # Interpolate
        interpolated_signal = interpolator((mesh_x, mesh_y, mesh_z))
        return interpolated_signal

