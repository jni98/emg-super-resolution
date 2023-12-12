import os
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp2d
import torch
from data import Dataset


class Interpolation:
    def __init__(self):
        # Load data
        data = pd.read_csv('../testset/data.csv', sep=";")
        self.testing = torch.utils.data.DataLoader(Dataset(data, "test"))

        # Create folders for B-Spline and Bicubic interpolated signals
        self.bspline_folder = 'interpolated_data/bspline'
        self.bicubic_folder = 'interpolated_data/bicubic'

    def b_spline_interpolation(self, lr_emg, hr_emg_shape):
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

    def bicubic_interpolation(self):
        pass

    def interpolate(self):
        for batch_idx, (lr_emg, hr_emg, lr_emg_paths, hr_emg_paths) in enumerate(self.testing):
            # 2D to 3D: reshape back to grid shape
            lr_emg = lr_emg.squeeze().numpy()
            lr_emg = lr_emg.transpose()
            lr_emg_reshaped = lr_emg.reshape(4, 4, lr_emg.shape[1])
            # Transpose and flip to get to correct grid numbering
            lr_emg_transposed = np.transpose(lr_emg_reshaped, axes=(1, 0, 2))
            lr_emg_grid = np.flipud(lr_emg_transposed)

            # 2D to 3D: reshape back to grid shape
            hr_emg = hr_emg.squeeze().numpy()
            hr_emg = hr_emg.transpose()
            hr_emg_reshaped = hr_emg.reshape(8, 8, hr_emg.shape[1])
            # Transpose and flip to get to correct grid numbering
            hr_emg_transposed = np.transpose(hr_emg_reshaped, axes=(1, 0, 2))
            hr_emg_grid = np.flipud(hr_emg_transposed)

            # Perform B-Spline interpolation
            interpolated_lr_hr = self.b_spline_interpolation(lr_emg_grid, (8, 8, lr_emg.shape[1]))
            interpolated_hr_sr = self.b_spline_interpolation(hr_emg_grid, (16, 16, lr_emg.shape[1]))

            # 3D to 2D: reshape back
            interpolated_lr_hr = np.flipud(interpolated_lr_hr).transpose(1, 0, 2).reshape(64, 512)
            interpolated_hr_sr = np.flipud(interpolated_hr_sr).transpose(1, 0, 2).reshape(256, 512)

            # flip rows and columns
            interpolated_lr_hr = interpolated_lr_hr.transpose()
            interpolated_hr_sr = interpolated_hr_sr.transpose()

            # Save B-Spline interpolated signal from lr to hr
            last_slash_index_lr = lr_emg_paths[0].rfind("/")
            bspline_path = lr_emg_paths[0][:last_slash_index_lr] + '/interpolated/bspline'
            os.makedirs(bspline_path, exist_ok=True)
            np.save(bspline_path + (lr_emg_paths[0][last_slash_index_lr:]).replace('-lr.npy', '-lr-sr.npy'),
                    interpolated_lr_hr)

            # Save B-Spline interpolated signal from hr to sr
            last_slash_index_hr = hr_emg_paths[0].rfind("/")
            np.save(bspline_path + (hr_emg_paths[0][last_slash_index_hr:]).replace('-hr.npy', '-hr-sr.npy'),
                    interpolated_hr_sr)

            # TODO: Perform bicubic interpolation


if __name__ == '__main__':
    interpolater = Interpolation()
    interpolater.interpolate()
