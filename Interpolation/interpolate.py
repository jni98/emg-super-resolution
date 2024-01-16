import json
import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import torch
from data import Custom_Dataset


class Interpolation:
    def __init__(self):
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

    def cubic_interpolation(self, lr_emg, hr_emg_shape):
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
        interpolator = RegularGridInterpolator((x, y, z), lr_emg)

        # Interpolate
        interpolated_signal = interpolator((mesh_x, mesh_y, mesh_z))

        return interpolated_signal

    def interpolate(self):
        for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
            # 2D to 3D: reshape back to grid shape
            lr_emg = lr_emg.squeeze().numpy()
            lr_emg = lr_emg.transpose()
            lr_emg_reshaped = lr_emg.reshape(4, 4, lr_emg.shape[1])
            # Transpose and flip to get to correct grid numbering
            lr_emg_transposed = np.transpose(lr_emg_reshaped, axes=(1, 0, 2))
            lr_emg_grid = np.flipud(lr_emg_transposed)

            # Perform cubic interpolation
            interpolated_spline_lr_hr = self.cubic_interpolation(lr_emg_grid, (7, 7, lr_emg.shape[1]))

            # 3D to 2D: reshape back
            interpolated_spline_lr_hr = np.flipud(interpolated_spline_lr_hr).transpose(1, 0, 2).reshape(49, 512)

            # flip rows and columns
            interpolated_spline_lr_hr = interpolated_spline_lr_hr.transpose()
            interpolated_spline_lr_hr = self.destandardize_data(interpolated_spline_lr_hr)

            # Save cubic interpolated signal from lr to hr
            last_slash_index_lr = lr_emg_paths[0].rfind("/")
            cubic_path = lr_emg_paths[0][:last_slash_index_lr] + '/interpolated/cubic'

            os.makedirs(cubic_path, exist_ok=True)
            np.save(cubic_path + (lr_emg_paths[0][last_slash_index_lr:]).replace('-lr.npy', '-lr-sr.npy'),
                    interpolated_spline_lr_hr)

    def destandardize_data(self, standardized_data):
        return standardized_data * self.mean_std_dict['std'] + self.mean_std_dict['mean']


if __name__ == '__main__':
    interpolater = Interpolation()
    interpolater.interpolate()
