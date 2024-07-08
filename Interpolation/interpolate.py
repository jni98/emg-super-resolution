"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Interpolation, which performs bicubic spline and bilinear interpolation
 on the low resolution data to get high resolution data.

You can run this code by creating an object of the class Interpolation and calling the interpolate method.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import torch
from data import Custom_Dataset
import scipy.ndimage
from sklearn.metrics import mean_squared_error

class Interpolation:
    """
    Class that performs bilinear and bicubic spline interpolation on the low resolution data to get high resolution data.
    """

    def __init__(self, dataset: str) -> None:
        """
        Constructs all the necessary attributes for the interpolation object.
        """

        # Load the JSON file with values for min max normalization into a Python dictionary
        with open(dataset + 'train/mean_std.json', 'r') as file:
            self.mean_std_dict = json.load(file)

        # Load data
        data = pd.read_csv(dataset + 'test/data.csv', sep=";")
        self.testing = torch.utils.data.DataLoader(
            Custom_Dataset(data, dataset + 'test', self.mean_std_dict['mean'], self.mean_std_dict['std']))

    def grid_interpolation(self, lr_emg: np.ndarray, hr_emg_shape: tuple, method: str) -> np.ndarray:
        """
        Performs bilienar interpolation on the low resolution data to get high resolution data.
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
        print("Created the interpolater")

        # Interpolate
        interpolated_signal = interpolator((mesh_x, mesh_y, mesh_z))
        print("Finished the interpolator")

        return interpolated_signal

    def interpolate_13x5(self) -> None:
        """
        Performs bicubic spline and bilinear interpolation on the low resolution data to get high resolution data,
        and calculates the mean squared error for each interpolation method. Saves the MSE to a file.
        """
        # Create variables for test loss using mse
        total_mse_bilinear = 0
        total_mse_bicubic = 0
        total_batches = len(self.testing)

        # Load mean and std for standardization before calculating test loss
        mean = self.mean_std_dict['mean']
        std = self.mean_std_dict['std']

        for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
            print(f'Batch index: {batch_idx}')

            lr_emg = lr_emg.numpy().squeeze()
            hr_emg = hr_emg.numpy().squeeze()

            # Perform bilinear interpolation using grid interpolation method
            interpolated_lr_hr = self.grid_interpolation(lr_emg, (13, 5, lr_emg.shape[2]), 'linear')

            # Save linear interpolated signal from lr to hr
            path = '../results/linear_Interpolation/'
            os.makedirs(path, exist_ok=True)
            print("Saving")
            np.save(path + lr_emg_paths[0], interpolated_lr_hr)

            # Standardize the data
            standardized_hr_emg = (hr_emg - mean) / std
            standardized_interpolated_lr_hr = (interpolated_lr_hr - mean) / std

            # Calculate and accumulate the MSE for bilinear interpolation
            mse_bilinear = mean_squared_error(standardized_hr_emg.flatten(), standardized_interpolated_lr_hr.flatten())
            total_mse_bilinear += mse_bilinear

            # Upsampling factors for the first two dimensions, keeping the third dimension the same
            zoom_factors = (13 / 7, 5 / 3, 1)

            # Perform bicubic spline interpolation
            interpolated_lr_hr = scipy.ndimage.zoom(lr_emg, zoom_factors, order=3)  # Using cubic interpolation

            # Save cubic interpolated signal from lr to hr
            path = '../results/cubic_interpolation/'
            os.makedirs(path, exist_ok=True)
            print("Saving")
            np.save(path + lr_emg_paths[0], interpolated_lr_hr)

            # Standardize the cubic interpolated data
            standardized_interpolated_lr_hr = (interpolated_lr_hr - mean) / std

            # Calculate and accumulate the MSE for cubic interpolation
            mse_bicubic = mean_squared_error(standardized_hr_emg.flatten(), standardized_interpolated_lr_hr.flatten())
            total_mse_bicubic += mse_bicubic

            # Save linear interpolated signal from lr to hr
            path = '../results/linear_zoom_interpolation/'
            os.makedirs(path, exist_ok=True)
            print("Saving")
            np.save(path + lr_emg_paths[0], interpolated_lr_hr)

        # Calculate average MSE
        average_mse_bilinear = total_mse_bilinear / total_batches
        average_mse_bicubic = total_mse_bicubic / total_batches

        print(f'Average MSE for linear interpolation: {average_mse_bilinear}')
        print(f'Average MSE for cubic interpolation: {average_mse_bicubic}')

        # Save the test loss to a file
        with open('test_loss.txt', 'w') as f:
            f.write(f'Average MSE for bilinear interpolation: {average_mse_bilinear}\n')
            f.write(f'Average MSE for bicubic interpolation: {average_mse_bicubic}\n')

if __name__ == '__main__':
    dataset = '../dataset/'
    interpolator = Interpolation(dataset)
    interpolator.interpolate_13x5()
