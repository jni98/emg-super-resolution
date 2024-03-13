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
        interpolation_lr_sr = np.load(self.file_path + 'linear_Interpolation/path0-lr.npy')
        model1_lr_sr = np.load(self.file_path + 'model_1/path0-lr.npy')
        model2_lr_sr = np.load(self.file_path + 'model_2/path0-lr.npy')

        # Load original data and normalize it
        origin_hr = np.load(self.file_path + 'groundtruth/path0-hr.npy')

        # Create a dictionary with titles as keys and EMG data as values
        self.emg_data_dict = {
            'Linear LR to SR': interpolation_lr_sr,
            'Model 1 LR to SR': model1_lr_sr,
            'Model 2 LR to SR': model2_lr_sr,
            'Original HR': origin_hr
        }

        print("init done")

    def plot_emg_channels(self) -> None:
        """
        Plot all emg channels for the original signal, the model 2 signal and the interpolated signal. Channels are
        plotted by an offset of 500.
        Returns
        -------
        None
        """

        print("Start plotting channels")

        time_axis = np.arange(64584)  # Assuming the time axis runs along axis 2

        for title, emg_data in self.emg_data_dict.items():
            # Plot Training losses in a separate figure
            plt.figure(figsize=(8.27, 14))

            offset = 0

            for rows in range(13):
                for cols in range(5):
                    signal = emg_data[rows][cols][:64584]

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
        Plot the mean absolute difference between the channels of the groundtruth signal and model 2 or interpolated
        signal.
        Results are displayed on the grid.

        Returns
        -------

        """
        print("Start plotting EMG differences")

        time_axis = np.arange(64584)  # Assuming the time axis runs along axis 2

        groundtruth = self.emg_data_dict['Original HR']

        for title, emg_data in self.emg_data_dict.items():
            # Plot Training losses in a separate figure
            plt.figure(figsize=(8.27, 14))

            groundtruth_cut = groundtruth[:, :, :64584]
            emg_data_cut = emg_data[:, :, :64584]

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
            plt.imshow(mean_absolute_differences, cmap='viridis', origin='upper', vmin=0, vmax=60)

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
        Plots a heatmap at one specific point in time for each the Original, the Linear interpolated and the model 2
        created signal.

        Returns
        -------

        """
        print("Start plotting heat maps")

        grids = (
            self.emg_data_dict['Original HR'],
            self.emg_data_dict['Linear LR to SR'],
            self.emg_data_dict['Model 1 LR to SR'],
            self.emg_data_dict['Model 2 LR to SR']
        )

        keys_list = ('Original HR', 'Linear', 'Model 1', 'Model 2')

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
        Plots the cross correlation and the dtw of the original, linear interpolated and model 2 signal.

        Returns
        -------

        """
        print("Start plotting correlations")

        grids = (
            self.emg_data_dict['Original HR'],
            self.emg_data_dict['Linear LR to SR'],
            self.emg_data_dict['Model 1 LR to SR'],
            self.emg_data_dict['Model 2 LR to SR'])

        # Accessing keys by index
        keys_list = ('Original HR', 'Linear LR to SR', 'Model 1 LR to SR', 'Model 2 LR to SR')

        for g in range(1, len(grids)):
            groundtruth = grids[0][:][:][:64584]
            grid_2 = grids[g][:][:][:64584]

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

    def plot_pearson_coefficient_grid(self) -> None:

        """
        Plots the pearson coefficient of the correlation of one channel in the groundtruth signal with all other
        channels of the interpolated or model 2 signal channels. Results are displayed on the grid.

        Returns
        -------

        """
        print("Start plotting pearson coefficient grids")

        grids = (
            self.emg_data_dict['Original HR'],
            self.emg_data_dict['Linear LR to SR'],
            self.emg_data_dict['Model 1 LR to SR'],
            self.emg_data_dict['Model 2 LR to SR'])

        # Accessing keys by index
        keys_list = ('Original HR', 'Linear LR to SR', 'Model 1 LR to SR', 'Model 2 LR to SR')

        for g in range(1, len(grids)):
            groundtruth = grids[0]
            grid_2 = grids[g]

            # Initialize an array to store correlation coefficients
            correlation_coefficients = np.zeros((13, 5))

            # Calculate Pearson correlation coefficients
            for row in range(13):
                for col in range(5):
                    correlation_coefficients[row, col], _ = pearsonr(groundtruth[row, col], grid_2[row, col])

            # Plot and save correlation grid
            plt.figure()
            plt.imshow(correlation_coefficients, cmap='viridis', origin='upper', vmin=0.95, vmax=1)

            # Set ticks and tick labels before calling imshow
            plt.xticks(range(5), labels=range(1, 6))
            plt.yticks(range(13), labels=range(13, 0, -1))

            plt.title(f'Pearson coefficient - Origin with {keys_list[g]}')
            plt.colorbar()

            # Create the directory if it doesn't exist
            directory = f'plot_pearsonr_heatmap/'
            os.makedirs(directory, exist_ok=True)

            plt.savefig(f'{directory}{keys_list[g]}.png')
            plt.close()

            # Correlation of one channel in the groundtruth signal with all other channels of the interpolated or model 2 signal channels
            for row1 in range(13):
                for col1 in range(5):
                    for row2 in range(13):
                        for col2 in range(5):
                            correlation_coefficients[row2, col2], _ = pearsonr(groundtruth[row1, col1],
                                                                               grid_2[row2, col2])

                    # Plot and save correlation grid
                    plt.figure()
                    plt.imshow(correlation_coefficients, cmap='viridis', origin='upper', vmin=0.3, vmax=1)

                    # Set ticks and tick labels before calling imshow
                    plt.xticks(range(5), labels=range(1, 6))
                    plt.yticks(range(13), labels=range(13, 0, -1))

                    plt.title(f'Pearson coefficient - Origin channel ({row1, col1}) with {keys_list[g]}')
                    plt.colorbar()

                    # Create the directory if it doesn't exist
                    directory = f'plot_pearsonr_heatmap_oneChannel/'
                    os.makedirs(directory, exist_ok=True)

                    plt.savefig(f'{directory}{keys_list[g]}-r{row1}c{col1}.png')
                    plt.close()


if __name__ == '__main__':
    # TODO: Beschriftung mit (predicted) etc. passt nicht zu HR to SR plots
    file_path = '../results/'

    visualize = Visualizer(file_path)

    visualize.plot_emg_channels()
    visualize.plot_emg_channels_difference()
    visualize.plot_heat_maps()
    # visualize.plot_correlations()
    visualize.plot_pearson_coefficient_grid()
