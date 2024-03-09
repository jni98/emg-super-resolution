"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Visualizer, which is used to visualize the results of the interpolation
and the neural networks.

The code is started by calling the main function that creates an object of the Visualizer class. The object is then used
to call the different visualization functions.
"""

import os

import numpy as np
import matplotlib
from dtw import accelerated_dtw

matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt

from scipy.stats import pearsonr


class Visualizer:
    """
    Class to visualize the results of the interpolation and the neural networks.
    """

    def __init__(self, file_path: str) -> None:
        """
        Constructs all the necessary attributes for the Visualizer object.
        Parameters
        ----------
        file_path: str
                      Path to the directory where the data is stored.
        """
        self.file_path = file_path

        # TODO: Load the correct data from the file_path

        # Load your data (already normalized)
        inter_spline_lr_sr = np.load(
            self.file_path + 'linear_Interpolation/path0-lr.npy')
        model1_lr_sr = np.load(self.file_path + 'model 2/path0-lr.npy')

        # Load original data and normalize it
        origin_hr = np.load(self.file_path + 'groundtruth/path0-hr.npy')

        # Create a dictionary with titles as keys and EMG data as values
        self.emg_data_dict = {
            'Linear LR to SR': inter_spline_lr_sr,
            'Model 1 LR to SR': model1_lr_sr,
            'Original HR': origin_hr
        }

        print("init done")

    def plot_emg_channels(self) -> None:
        """
        Plot all emg channels for the original signal, the model 1 signal and the interpolated signal. Channels are
        plotted by an offset of 500.
        Returns
        -------
        None
        """

        print("Start plotting channels")

        time_axis = np.arange(60754)  # Assuming the time axis runs along axis 2

        for title, emg_data in self.emg_data_dict.items():
            # Plot Training losses in a separate figure
            plt.figure(figsize=(8.27, 14))

            offset = 0

            for rows in range(13):
                for cols in range(5):
                    signal = emg_data[rows][cols][:60754]

                    predicted = ''
                    # if title == 'Original HR':
                    #     predicted = ''
                    #
                    # else:
                    #     if (j % 2) != 0:
                    #         predicted = '(predicted)'
                    #     elif (j > 6 and j < 14) or (j > 20 and j < 28) or (j > 34 and j < 42):
                    #         predicted = '(predicted)'
                    #     else:
                    #         predicted = ''

                    plt.plot(time_axis, signal.squeeze() + offset,
                             label=f'Channel row{rows + 1} col{cols + 1} {predicted}')
                    offset += 3500

            plt.xlabel('Time in samples')
            plt.title(title)

            # Retrieve handles and labels for legend
            handles, labels = plt.gca().get_legend_handles_labels()

            # Reverse the order of handles and labels
            handles = handles[::-1]
            labels = labels[::-1]

            plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()

            # Create the directory if it doesn't exist
            directory = f'plot_channels/'
            os.makedirs(directory, exist_ok=True)

            plt.savefig(f'{directory}{title}.png')
            plt.close()
            plt.figure()

    def plot_emg_channels_difference(self) -> None:
        """
        Plot the mean absolute difference between the channels of the groundtruth signal and model 1 or interpolated
        signal.
        Results are displayed on the grid.

        Returns
        -------

        """
        print("Start plotting EMG differences")

        time_axis = np.arange(60754)  # Assuming the time axis runs along axis 2

        groundtruth = self.emg_data_dict['Original HR']

        for title, emg_data in self.emg_data_dict.items():
            # Plot Training losses in a separate figure
            plt.figure(figsize=(8.27, 14))

            groundtruth_cut = groundtruth[:, :, :60754]
            emg_data_cut = emg_data[:, :, :60754]

            mean_absolute_differences = np.mean(np.abs(groundtruth_cut - emg_data_cut), axis=2)

            plt.xlabel('Time in samples')
            plt.title(title)

            # Retrieve handles and labels for legend
            handles, labels = plt.gca().get_legend_handles_labels()

            # Reverse the order of handles and labels
            handles = handles[::-1]
            labels = labels[::-1]

            plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))

            plt.tight_layout()  # Adjust layouts to avoid overlapping

            # Create the directory if it doesn't exist
            directory = f'plot_channels_difference/'
            os.makedirs(directory, exist_ok=True)

            plt.savefig(f'{directory}{title}.png')
            plt.close()
            plt.figure()

            # Plot and save mean_differences_grid
            plt.figure()
            plt.imshow(mean_absolute_differences, cmap='viridis', origin='upper', vmin=0, vmax=100)

            # Set ticks and tick labels before calling imshow
            plt.xticks(range(5), labels=range(1, 6))
            plt.yticks(range(13), labels=range(13, 0, -1))

            plt.title(f'Mean absolute differences - {title}')
            plt.colorbar()

            # Create the directory if it doesn't exist
            directory = f'plot_mean_differences_heatmap/'
            os.makedirs(directory, exist_ok=True)

            plt.savefig(f'{directory}{title}_mean_differences.png')
            plt.close()

    def plot_heat_maps(self) -> None:
        """
        Plots a heatmap at one specific point in time for each the Original, the Linear interpolated and the model 1
        created signal.

        Returns
        -------

        """
        print("Start plotting heat maps")

        grids = (
            self.emg_data_dict['Original HR'],
            self.emg_data_dict['Linear LR to SR'],
            self.emg_data_dict['Model 1 LR to SR']
        )

        keys_list = ('Original HR', 'Linear', 'Model 1')

        time_points = [20000, 30000, 40000]

        fig, axs = plt.subplots(len(grids), len(time_points))

        for g in range(len(grids)):
            emg_data = grids[g]

            for i, time_point in enumerate(time_points):
                emg_heatmap = emg_data[:, :, time_point]
                im = axs[g][i].imshow(emg_heatmap, cmap='hot')

                # Set ticks and tick labels to start from 1 but position them at 0 to match the data
                axs[g][i].set_xticks(range(5))
                axs[g][i].set_yticks(range(13))
                axs[g][i].set_xticklabels(range(1, 6))
                axs[g][i].set_yticklabels(range(1, 14))
                axs[g][i].set_title(keys_list[g] + f'-{time_point}')

                axs[g][i].invert_yaxis()  # Invert the y-axis direction
                fig.colorbar(im, ax=axs[g][i])

        plt.tight_layout()  # Adjust layouts to avoid overlapping

        # Create the directory if it doesn't exist
        directory = f'plot_heatmaps/'
        os.makedirs(directory, exist_ok=True)

        plt.savefig(f'{directory}Heatmaps.png')

    def plot_correlations(self) -> None:
        """
        Plots the cross correlation and the dtw of the original, linear interpolated and model 1 signal.

        Returns
        -------

        """
        print("Start plotting correlations")

        grids = (
            self.emg_data_dict['Original HR'], self.emg_data_dict['Linear LR to SR'],
            self.emg_data_dict['Model 1 LR to SR'])

        # Accessing keys by index
        keys_list = ('Original HR', 'Linear LR to SR', 'Model 1 LR to SR')

        for g in range(1, len(grids)):
            groundtruth = grids[0][:][:][:60754]
            grid_2 = grids[g][:][:][:60754]

            plt.figure()
            number_graphs = 0
            plot_number = 1

            for row in range(13):
                for col in range(5):
                    # Iterate through electrodes
                    correlation = np.correlate(groundtruth[row][col], grid_2[row][col], mode='full')

                    # Calculate autocorrelation of groundtruth
                    autocorr_groundtruth = np.correlate(groundtruth[row][col], groundtruth[row][col], mode='full')

                    # normalize correlation by autocorrelation
                    if (row == 12 and col == 0):
                        normalized_correlation = correlation
                    else:
                        normalized_correlation = correlation / np.max(autocorr_groundtruth)

                    # Find the maximum and minimum values of normalized_corr across all electrodes
                    # max_corr = max([max(corr) for corr in normalized_corr])
                    # min_corr = min([min(corr) for corr in normalized_corr])

                    # TODO: Achtung, predicted stimmt nicht mehr
                    predicted = ''

                    plt.plot(normalized_correlation, label=f'Ch row {row} col {col} {predicted}')
                    number_graphs += 1
                    plt.xlabel('Shift')
                    plt.ylabel('Correlation')
                    plt.legend(loc='best', bbox_to_anchor=(1, 1))
                    plt.title(f'Cross-correlation of {keys_list[g]} with Original HR - {plot_number}')

                    # TODO: Adjust limits
                    # Set the y-axis limits to the maximum and minimum values across all electrodes
                    plt.ylim(-0.4, 1.2)

                    plt.tight_layout()  # Adjust layouts to avoid overlapping

                    if number_graphs % 5 == 0:
                        # Create the directory if it doesn't exist
                        directory = f'plot_correlations/{keys_list[g]}/'
                        os.makedirs(directory, exist_ok=True)

                        plt.savefig(f'{directory}{keys_list[g]} with Original HR - {plot_number}.png')
                        plt.close()
                        plt.figure()
                        plot_number += 1

    def plot_pearson_coefficient(self) -> None:

        """
        Plots the pearson coefficient of the original signal with the linear interpolated or the model 1 created signal.
        Each of the 49 channels is correlated with each of the artifical 49 channels.

        Returns
        -------

        """

        print("Start plotting pearson coefficient")

        grids = (
            self.emg_data_dict['Original HR'], self.emg_data_dict['Linear LR to SR'],
            self.emg_data_dict['Model 1 LR to SR'])

        # Accessing keys by index
        keys_list = ('Original HR', 'Linear LR to SR', 'Model 1 LR to SR')

        for g in range(1, len(grids)):
            groundtruth = grids[0]
            groundtruth_shape = groundtruth.shape
            grid_2 = grids[g]
            grid_2_shape = grid_2.shape

            # Initialize an array to store correlation coefficients
            correlation_coefficients = np.zeros((65, 65))

            # 3D to 2D
            groundtruth[:, 1::2, :] = np.flipud(groundtruth[:, 1::2, :])
            groundtruth = np.flipud(groundtruth).transpose(1, 0, 2).reshape(65, groundtruth_shape[2])
            groundtruth_cut = groundtruth[:65, :60754]

            # 3D to 2D
            grid_2[:, 1::2, :] = np.flipud(grid_2[:, 1::2, :])
            grid_2 = np.flipud(grid_2).transpose(1, 0, 2).reshape(65, grid_2_shape[2])
            grid_2_cut = grid_2[:65, :60754]

            # Calculate Pearson correlation coefficients
            for i in range(1, 65):
                for j in range(1, 65):
                    # TODO: Muss sinnvoll umgeschrieben werde
                    correlation_coefficients[i, j], _ = pearsonr(groundtruth_cut[i], grid_2_cut[j])

            # Mask the upper triangle
            mask = np.tril(np.ones_like(correlation_coefficients, dtype=bool))

            # Create a larger heatmap
            plt.figure(figsize=(8.27, 7))

            # Plot the heatmap with values and ticks
            plt.imshow(np.ma.masked_array(correlation_coefficients, ~mask), cmap='viridis', origin='upper', vmin=0,
                       vmax=1)

            # Add colorbar
            cbar = plt.colorbar()
            cbar.set_label('Pearson Coefficient')

            # Add ticks to both axes
            plt.xticks(np.arange(0, 64, step=5), labels=np.arange(0, 64, step=5))
            plt.yticks(np.arange(0, 64, step=5), labels=np.arange(0, 64, step=5))

            plt.title(f'Pearson coefficient of {keys_list[g]} with Original HR')
            plt.xlabel(f'{keys_list[g]}')
            plt.ylabel('Original HR')
            plt.tight_layout()

            # Create the directory if it doesn't exist
            directory = f'plot_pearsonr/'
            os.makedirs(directory, exist_ok=True)

            plt.savefig(f'{directory}{keys_list[g]} with Original HR.png')

    def plot_pearson_coefficient_grid(self) -> None:

        """
        Plots the pearson coefficient of the correlation of one channel in the groundtruth signal with all other
        channels of the interpolated or model 1 signal channels. Results are displayed on the grid.

        Returns
        -------

        """
        print("Start plotting pearson coefficient grids")

        grids = (
            self.emg_data_dict['Original HR'], self.emg_data_dict['Linear LR to SR'],
            self.emg_data_dict['Model 1 LR to SR'])

        # Accessing keys by index
        keys_list = ('Original HR', 'Linear LR to SR', 'Model 1 LR to SR')

        for g in range(1, len(grids)):
            groundtruth = grids[0]
            groundtruth_shape = groundtruth.shape
            grid_2 = grids[g]
            grid_2_shape = grid_2.shape

            # 3D to 2D
            groundtruth[:, 1::2, :] = np.flipud(groundtruth[:, 1::2, :])
            groundtruth = np.flipud(groundtruth).transpose(1, 0, 2).reshape(65, groundtruth_shape[2])
            groundtruth_cut = groundtruth[:65, :60754]

            # 3D to 2D
            grid_2[:, 1::2, :] = np.flipud(grid_2[:, 1::2, :])
            grid_2 = np.flipud(grid_2).transpose(1, 0, 2).reshape(65, grid_2_shape[2])
            grid_2_cut = grid_2[:65, :60754]

            # Initialize an array to store correlation coefficients
            correlation_coefficients = np.zeros((65, 65))

            # Initialize an array to store correlation coefficients
            correlation_coefficients_diagonal = np.zeros(65)

            # Calculate Pearson correlation coefficients
            for i in range(1, 65):
                for j in range(1, 65):
                    correlation_coefficients[i, j], _ = pearsonr(groundtruth_cut[i], grid_2_cut[j])

                correlation_grid = correlation_coefficients[i]

                # 2D to 3D:
                correlation_grid = correlation_grid.reshape(5, 13).transpose(1, 0)
                correlation_grid = np.flipud(correlation_grid)
                correlation_grid[:, 1::2] = np.flipud(correlation_grid[:, 1::2])

                # Plot and save mean_differences_grid
                plt.figure()
                plt.imshow(correlation_grid, cmap='viridis', origin='upper', vmin=0.2, vmax=1)

                # Set ticks and tick labels before calling imshow
                plt.xticks(range(5), labels=range(1, 6))
                plt.yticks(range(13), labels=range(13, 0, -1))

                plt.title(f'Pearson coefficient - Original channel {i + 1} with {keys_list[g]}')
                plt.colorbar()

                # Create the directory if it doesn't exist
                directory = f'plot_pearsonr_heatmap/'
                os.makedirs(directory, exist_ok=True)

                plt.savefig(f'{directory}OriginalChannel{i + 1}_{keys_list[g]}.png')
                plt.close()

            # Calculate Pearson correlation coefficients
            for i in range(65):
                correlation_coefficients_diagonal[i], _ = pearsonr(groundtruth_cut[i], grid_2_cut[i])

            # 2D to 3D:
            correlation_coefficients_diagonal = correlation_coefficients_diagonal.reshape(5, 13).transpose(1, 0)
            correlation_coefficients_diagonal = np.flipud(correlation_coefficients_diagonal)
            correlation_coefficients_diagonal[:, 1::2] = np.flipud(correlation_coefficients_diagonal[:, 1::2])

            # Plot and save mean_differences_grid
            plt.figure()
            plt.imshow(correlation_coefficients_diagonal, cmap='viridis', origin='upper', vmin=0.85, vmax=1)

            # Set ticks and tick labels before calling imshow
            plt.xticks(range(5), labels=range(1, 6))
            plt.yticks(range(13), labels=range(13, 0, -1))

            plt.title(f'Pearson coefficient - Diagonal - Origin with {keys_list[g]}')
            plt.colorbar()

            # Create the directory if it doesn't exist
            directory = f'plot_pearsonr_heatmap_diagonal/'
            os.makedirs(directory, exist_ok=True)

            plt.savefig(f'{directory}Diagonal_{keys_list[g]}.png')
            plt.close()

            # Mask the upper triangle
            mask = np.tril(np.ones_like(correlation_coefficients, dtype=bool))

            # Create a larger heatmap
            plt.figure(figsize=(8.27, 7))

            # Plot the heatmap with values and ticks
            plt.imshow(np.ma.masked_array(correlation_coefficients, ~mask), cmap='viridis', origin='upper', vmin=0,
                       vmax=1)

            # Add colorbar
            cbar = plt.colorbar()
            cbar.set_label('Pearson Coefficient')

            # Add ticks to both axes
            plt.xticks(np.arange(0, 65, step=5), labels=np.arange(0, 65, step=5))
            plt.yticks(np.arange(0, 65, step=5), labels=np.arange(0, 65, step=5))

            plt.title(f'Pearson coefficient of {keys_list[g]} with Original HR')
            plt.xlabel(f'{keys_list[g]}')
            plt.ylabel('Original HR')
            plt.tight_layout()

            # Create the directory if it doesn't exist
            directory = f'plot_pearsonr/'
            os.makedirs(directory, exist_ok=True)

            plt.savefig(f'{directory}{keys_list[g]} with Original HR.png')


if __name__ == '__main__':
    # TODO: Beschriftung mit (predicted) etc. passt nicht zu HR to SR plots
    file_path = '../results/'

    visualize = Visualizer(file_path)

    visualize.plot_emg_channels()
    visualize.plot_emg_channels_difference()
    visualize.plot_heat_maps()
    visualize.plot_correlations()
    visualize.plot_pearson_coefficient()
    visualize.plot_pearson_coefficient_grid()
