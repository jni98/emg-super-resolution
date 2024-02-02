"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Interpolation, which performs cubic interpolation on the low resolution
data to get high resolution data.

You can run this code by creating an object of the class Interpolation and calling the interpolate method.
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import torch
from data import Custom_Dataset


class Interpolation:
    """
    Class that performs cubic interpolation on the low resolution data to get high resolution data.
    """

    def __init__(self) -> None:
        """
        Constructs all the necessary attributes for the interpolation object.
        """
        # Load the JSON file with values for min max normalization into a Python dictionary
        with open('../dataset/mean_std.json', 'r') as file:
            self.mean_std_dict = json.load(file)

        # Load the JSON file with values for min max normalization into a Python dictionary
        with open('../dataset/min_max.json', 'r') as file:
            min_max_dict = json.load(file)

        # Load data
        data = pd.read_csv('../testset/data.csv', sep=";")
        self.testing = torch.utils.data.DataLoader(
            Custom_Dataset(data, min_max_dict['min'], min_max_dict['max'], self.mean_std_dict['mean'],
                           self.mean_std_dict['std']))

        #TODO: Add linear and b-spline interpolation

    def cubic_interpolation(self, lr_emg: np.ndarray, hr_emg_shape: tuple) -> np.ndarray:
        """
        Performs cubic interpolation on the low resolution data to get high resolution data.
        Parameters
        ----------
        lr_emg: numpy.ndarray
                Low resolution emg data.
        hr_emg_shape: tuple
                Shape of the high resolution emg data.

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
        interpolator = RegularGridInterpolator((x, y, z), lr_emg, method='cubic')
        print("Created the interpolater")

        # Interpolate
        interpolated_signal = interpolator((mesh_x, mesh_y, mesh_z))
        print("Finished the interpolator")
        return interpolated_signal

    def interpolate_8x8(self) -> None:
        """
        Performs cubic interpolation on the low resolution data to get high resolution data.

        Returns
        -------
        interpolated_lr_hr: numpy.ndarray
                High resolution emg data.
        """
        for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
            print(f'{batch_idx} of 157 batches')
            # 2D to 3D: reshape back to grid shape
            lr_emg = lr_emg.squeeze().numpy()
            lr_emg = lr_emg.transpose()
            lr_emg_reshaped = lr_emg.reshape(4, 4, lr_emg.shape[1])

            # Transpose and flip to get to correct grid numbering
            lr_emg_transposed = np.transpose(lr_emg_reshaped, axes=(1, 0, 2))
            lr_emg_grid = np.flipud(lr_emg_transposed)
            lr_emg_grid = self.destandardize_data(lr_emg_grid)

            # Perform cubic interpolation
            interpolated_lr_hr = self.cubic_interpolation(lr_emg_grid, (7, 7, lr_emg.shape[1]))

            # 3D to 2D: reshape back
            interpolated_lr_hr = np.flipud(interpolated_lr_hr).transpose(1, 0, 2).reshape(49, 512)

            # flip rows and columns
            interpolated_lr_hr = interpolated_lr_hr.transpose()

            # Save cubic interpolated signal from lr to hr
            last_slash_index_lr = lr_emg_paths[0].rfind("/")
            cubic_path = lr_emg_paths[0][:last_slash_index_lr] + '/interpolated/cubic/8x8'

            os.makedirs(cubic_path, exist_ok=True)
            print("Saving")
            np.save(cubic_path + (lr_emg_paths[0][last_slash_index_lr:]).replace('-lr.npy', '-lr-sr.npy'),
                    interpolated_lr_hr)

    def interpolate_13x5(self) -> None:
        """
        Performs cubic interpolation on the low resolution data to get high resolution data.

        Returns
        -------
        interpolated_lr_hr: numpy.ndarray
                High resolution emg data.
        """
        for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
            print(f'{batch_idx} of 157 batches')
            # 2D to 3D: reshape back to grid shape
            lr_emg = lr_emg.squeeze().numpy()
            lr_emg = lr_emg.transpose()
            lr_emg_reshaped = lr_emg.reshape(7, 3, lr_emg.shape[1])

            # Transpose and flip to get to correct grid numbering
            lr_emg_transposed = np.transpose(lr_emg_reshaped, axes=(1, 0, 2))
            lr_emg_grid = np.flipud(lr_emg_transposed)
            lr_emg_grid = self.destandardize_data(lr_emg_grid)

            # Perform cubic interpolation
            interpolated_lr_hr = self.cubic_interpolation(lr_emg_grid, (13, 5, lr_emg.shape[1]))

            # 3D to 2D: reshape back
            interpolated_lr_hr = np.flipud(interpolated_lr_hr).transpose(1, 0, 2).reshape(64, 2048)

            # flip rows and columns
            interpolated_lr_hr = interpolated_lr_hr.transpose()

            # Save cubic interpolated signal from lr to hr
            last_slash_index_lr = lr_emg_paths[0].rfind("/")
            cubic_path = lr_emg_paths[0][:last_slash_index_lr] + '/interpolated/cubic/13x5'

            os.makedirs(cubic_path, exist_ok=True)
            print("Saving")
            np.save(cubic_path + (lr_emg_paths[0][last_slash_index_lr:]).replace('-lr.npy', '-lr-sr.npy'),
                    interpolated_lr_hr)

    def destandardize_data(self, standardized_data: np.ndarray) -> np.ndarray:
        """
        De-standardizes the data according to the mean and standard deviation of the training data.
        Parameters
        ----------
        standardized_data: numpy.ndarray
                Standardized data.

        Returns
        -------
        destandardized_data: numpy.ndarray
                De-standardized data.
        """
        return standardized_data * self.mean_std_dict['std'] + self.mean_std_dict['mean']


if __name__ == '__main__':
    interpolater = Interpolation()
    interpolater.interpolate_8x8()
