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

        # Load your data (already normalized)
        inter_spline_lr_sr = np.load(
            self.file_path + '/interpolated/cubic/grid-0-window5120-pos-0-0-lr-sr.npy').transpose()
        model1_lr_sr = np.load(self.file_path + '/model1/grid-0-window5120-pos-0-0-lr-sr.npy').transpose()
        model1_hr_sr = np.load(self.file_path + '/model1/grid-0-window5120-pos-0-0-hr-sr.npy').transpose()

        # Load original data and normalize it
        origin_hr = np.load(self.file_path + '/grid-0-window5120-pos-0-0-hr.npy').transpose()

        # Create a dictionary with titles as keys and EMG data as values
        self.emg_data_dict = {
            'Cubic LR to SR': inter_spline_lr_sr,
            'Model 1 LR to SR': model1_lr_sr,
            'Model 1 HR to SR': model1_hr_sr,
            'Original HR': origin_hr
        }

    def plot_emg_channels(self) -> None:
        """
        Plot all emg channels for the original signal, the model 1 signal and the interpolated signal. Channels are plotted by an offset of 500.
        Returns
        -------
        None
        """

        time_axis = np.arange(200)  # Assuming the time axis runs along axis 2

        for title, emg_data in self.emg_data_dict.items():
            # Plot Training losses in a separate figure
            plt.figure(figsize=(8.27, 14))

            offset = 0
            # Use only few channels here
            for j in range(49):
                # TODO: Achtung, nicht das volle Zeitfenster
                signal = emg_data[j][:200]

                if title == 'Original HR':
                    predicted = ''

                else:
                    if (j % 2) != 0:
                        predicted = '(predicted)'
                    elif (j > 6 and j < 14) or (j > 20 and j < 28) or (j > 34 and j < 42):
                        predicted = '(predicted)'
                    else:
                        predicted = ''

                plt.plot(time_axis, signal.squeeze() + offset, label=f'Channel {j + 1} {predicted}')
                offset += 500

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
        Plot the mean absolute difference between the channels of the groundtruth signal and model 1 or interpolated signal.
        Results are displayed on the grid.

        Returns
        -------

        """
        time_axis = np.arange(512)  # Assuming the time axis runs along axis 2

        groundtruth = self.emg_data_dict['Original HR']

        for title, emg_data in self.emg_data_dict.items():
            # Plot Training losses in a separate figure
            plt.figure(figsize=(8.27, 14))

            offset = 0

            mean_absolute_differences = np.zeros(49)

            # Use only few channels here
            for j in range(49):
                # TODO: Achtung, nicht das volle Zeitfenster
                signal = emg_data[j]
                groundtruth_ch = groundtruth[j]
                difference = groundtruth_ch.squeeze() - signal.squeeze()

                mean_absolute_difference = np.mean(np.abs(groundtruth_ch.squeeze() - signal.squeeze()))
                mean_absolute_differences[j] = mean_absolute_difference

                if title == 'Original HR':
                    predicted = ''

                else:
                    if (j % 2) != 0:
                        predicted = '(predicted)'
                    elif (j > 6 and j < 14) or (j > 20 and j < 28) or (j > 34 and j < 42):
                        predicted = '(predicted)'
                    else:
                        predicted = ''

                plt.plot(time_axis, (difference + offset), label=f'Channel {j + 1} {predicted}')
                offset += 400

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

            # 2D to 3D:
            # reshape back to grid shape
            mean_differences_reshaped = mean_absolute_differences.reshape(7, 7)
            # Transpose and flip to get to correct grid numbering
            mean_differences_transposed = np.transpose(mean_differences_reshaped, axes=(1, 0))
            mean_differences_grid = np.flipud(mean_differences_transposed)

            # Plot and save mean_differences_grid
            plt.figure()
            plt.imshow(mean_differences_grid, cmap='viridis', origin='upper', vmin=0, vmax=40)

            # Set ticks and tick labels before calling imshow
            plt.xticks(range(7), labels=range(1, 8))
            plt.yticks(range(7), labels=range(7, 0, -1))

            plt.title(f'Mean absolute differences - {title}')
            plt.colorbar()

            # Create the directory if it doesn't exist
            directory = f'plot_mean_differences_heatmap/'
            os.makedirs(directory, exist_ok=True)

            plt.savefig(f'{directory}{title}_mean_differences.png')
            plt.close()

    def plot_heat_maps(self) -> None:
        """
        Plots a heatmap at one specific point in time for each the Original, the cubic interpolated and the model 1 created signal.

        Returns
        -------

        """
        grids = (
            self.emg_data_dict['Original HR'],
            self.emg_data_dict['Cubic LR to SR'],
            self.emg_data_dict['Model 1 LR to SR']
        )

        keys_list = ('Original HR', 'Cubic', 'Model 1')

        time_points = [200, 300, 400]

        fig, axs = plt.subplots(len(grids), len(time_points))

        for g in range(len(grids)):
            emg_data = grids[g]
            emg_data = emg_data.transpose()
            emg_data_reshaped = emg_data.reshape(7, 7, 512)
            emg_data_transposed = np.transpose(emg_data_reshaped, axes=(1, 0, 2))
            emg_data_grid = np.flipud(emg_data_transposed)

            for i, time_point in enumerate(time_points):
                emg_heatmap = emg_data_grid[:, :, time_point]
                im = axs[g][i].imshow(emg_heatmap, cmap='hot')

                # Set ticks and tick labels to start from 1 but position them at 0 to match the data
                axs[g][i].set_xticks(range(7))
                axs[g][i].set_yticks(range(7))
                axs[g][i].set_xticklabels(range(1, 8))
                axs[g][i].set_yticklabels(range(1, 8))
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
        Plots the cross correlation and the dtw of the original, cubic interpolated and model 1 signal.

        Returns
        -------

        """
        grids = (
            self.emg_data_dict['Original HR'], self.emg_data_dict['Cubic LR to SR'],
            self.emg_data_dict['Model 1 LR to SR'])

        # Accessing keys by index
        keys_list = ('Original HR', 'Cubic LR to SR', 'Model 1 LR to SR')

        # Create a list of line styles, colors
        colors = ['blue', 'green', 'red', 'black']  # Add more colors if required
        line_styles = ['-', '--']  # You can extend this list as needed

        for g in range(1, len(grids)):
            groundtruth = grids[0]
            grid_2 = grids[g]

            # Calculate cross-correlation for specific pairs of electrodes
            normalized_corr = []
            plt.figure()

            for i in range(49):  # Iterate through electrodes
                correlation = np.correlate(groundtruth[i], grid_2[i], mode='full')

                # Calculate autocorrelation of groundtruth
                autocorr_groundtruth = np.correlate(groundtruth[i], groundtruth[i], mode='full')

                # normalize correlation by autocorrelation
                normalized_correlation = correlation / np.max(autocorr_groundtruth)

                normalized_corr.append(normalized_correlation)

                # Calculate dynamic time warping (DTW)
                d, cost_matrix, acc_cost_matrix, path = accelerated_dtw(groundtruth[i], grid_2[i], dist='euclidean')
                # Plot and save the DTW visualization for each electrode
                plt.imshow(acc_cost_matrix.T, origin='lower', cmap='gray', interpolation='nearest')
                plt.plot(path[0], path[1], 'w')

                # Save the plot with a unique filename for each electrode (e.g., using electrode index)
                # Create the directory if it doesn't exist
                directory = f'plot_dtw/{keys_list[g]}/'
                os.makedirs(directory, exist_ok=True)

                plt.title(f'DTW {keys_list[g]} - Electrode {i}')
                plt.xlabel('Ground Truth')
                plt.ylabel('Model Prediction')
                plt.savefig(f'{directory}dtw_visualization_electrode_{i}.png')  # Save the plot as an image
                plt.close()

            # Find the maximum and minimum values of normalized_corr across all electrodes
            max_corr = max([max(corr) for corr in normalized_corr])
            min_corr = min([min(corr) for corr in normalized_corr])

            # Plot the normalized_corr
            plt.figure()
            color = 0
            style = 0
            number = 1
            for i, corr in enumerate(normalized_corr):
                if ((i + 1) % 2) != 0:
                    predicted = '(predicted)'
                elif (i > 7 and i < 16) or (i > 23 and i < 32) or (i > 39 and i < 48):
                    predicted = '(predicted)'
                else:
                    predicted = ''

                plt.plot(corr, label=f'Ch {i + 1} {predicted}', linestyle=line_styles[style % len(line_styles)],
                         color=colors[color % len(colors)])  #
                plt.xlabel('Shift')
                plt.ylabel('Correlation')
                plt.legend(loc='best', bbox_to_anchor=(1, 1))
                plt.title(f'Cross-correlation of {keys_list[g]} with Original HR')

                #TODO: Set in a useful way
                # Set the y-axis limits to the maximum and minimum values across all electrodes
                plt.ylim(-0.4, 1.2)

                color += 1
                if color == 4:
                    color = 0
                    style += 1

                if ((i + 1) % 8) == 0:
                    plt.tight_layout()  # Adjust layouts to avoid overlapping

                    # Create the directory if it doesn't exist
                    directory = f'plot_correlations/{keys_list[g]}/'
                    os.makedirs(directory, exist_ok=True)

                    plt.savefig(f'{directory}{keys_list[g]} with Original HR - {number}.png')
                    number += 1
                    plt.close()
                    plt.figure()
                    color = 0
                    style = 0

    def plot_pearson_coefficient(self) -> None:

        """
        Plots the pearson coefficient of the original signal with the cubic interpolated or the model 1 created signal.
        Each of the 49 channels is correlated with each of the artifical 49 channels.

        Returns
        -------

        """

        grids = (
            self.emg_data_dict['Original HR'], self.emg_data_dict['Cubic LR to SR'],
            self.emg_data_dict['Model 1 LR to SR'])

        # Accessing keys by index
        keys_list = ('Original HR', 'Cubic LR to SR', 'Model 1 LR to SR')

        for g in range(1, len(grids)):
            groundtruth = grids[0]
            grid_2 = grids[g]

            # Initialize an array to store correlation coefficients
            correlation_coefficients = np.zeros((49, 49))

            # Calculate Pearson correlation coefficients
            for i in range(49):
                for j in range(49):
                    correlation_coefficients[i, j], _ = pearsonr(groundtruth[i], grid_2[j])

            # Mask the upper triangle
            mask = np.tril(np.ones_like(correlation_coefficients, dtype=bool))

            # Create a larger heatmap
            plt.figure(figsize=(8.27, 7))

            # Plot the heatmap with values and ticks
            plt.imshow(np.ma.masked_array(correlation_coefficients, ~mask), cmap='viridis', origin='upper', vmin=0,
                       vmax=1)

            # Add specific values to each cell
            # for i in range(49):
            #     for j in range(49):
            #         plt.text(j, i, f'{correlation_coefficients[i, j]:.2f}', ha='center', va='center', color='white')

            # Add colorbar
            cbar = plt.colorbar()
            cbar.set_label('Pearson Coefficient')

            # Add ticks to both axes
            plt.xticks(np.arange(0, 49, step=5), labels=np.arange(0, 49, step=5))
            plt.yticks(np.arange(0, 49, step=5), labels=np.arange(0, 49, step=5))

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
        Plots the pearson coefficient of the correlation of one channel in the groundtruth signal with all other channels
        of the interpolated or model 1 signal channels. Results are displayed on the grid.

        Returns
        -------

        """

        grids = (
            self.emg_data_dict['Original HR'], self.emg_data_dict['Cubic LR to SR'],
            self.emg_data_dict['Model 1 LR to SR'])

        # Accessing keys by index
        keys_list = ('Original HR', 'Cubic LR to SR', 'Model 1 LR to SR')

        for g in range(1, len(grids)):
            groundtruth = grids[0]
            grid_2 = grids[g]

            # Initialize an array to store correlation coefficients
            correlation_coefficients = np.zeros((49, 49))

            # Initialize an array to store correlation coefficients
            correlation_coefficients_diagonal = np.zeros((49))

            # Calculate Pearson correlation coefficients
            for i in range(49):
                for j in range(49):
                    correlation_coefficients[i, j], _ = pearsonr(groundtruth[i], grid_2[j])

                correlation_grid = correlation_coefficients[i]
                correlation_grid = correlation_grid.reshape(7, 7)
                correlation_grid = np.transpose(correlation_grid, axes=(1, 0))
                correlation_grid = np.flipud(correlation_grid)

                # Plot and save mean_differences_grid
                plt.figure()
                plt.imshow(correlation_grid, cmap='viridis', origin='upper', vmin=0.2, vmax=1)

                # Set ticks and tick labels before calling imshow
                plt.xticks(range(7), labels=range(1, 8))
                plt.yticks(range(7), labels=range(7, 0, -1))

                plt.title(f'Pearson coefficient - Original channel {i + 1} with {keys_list[g]}')
                plt.colorbar()

                # Create the directory if it doesn't exist
                directory = f'plot_pearsonr_heatmap/'
                os.makedirs(directory, exist_ok=True)

                plt.savefig(f'{directory}OriginalChannel{i + 1}_{keys_list[g]}.png')
                plt.close()

            # TODO: Perform this for all grids and add to 13x5
            if g == 1:
                correlation_grid_cubic = correlation_grid
            else:
                correlation_grid_model1 = correlation_grid

                # Subtract grids to get the difference
                difference_grid = correlation_grid_cubic - correlation_grid_model1

                # Plot and save the difference between the pearson coefficients of the two grids
                plt.figure()
                plt.imshow(difference_grid, cmap='viridis', origin='upper', vmin=-0.1, vmax=0.1)

                # Set ticks and tick labels before calling imshow
                plt.xticks(range(7), labels=range(1, 8))
                plt.yticks(range(7), labels=range(7, 0, -1))

                plt.title(f'Difference of Pearson coefficients - Original channel {i + 1}')
                plt.colorbar()

                # Create the directory if it doesn't exist
                directory = f'plot_pearsonr_difference/'
                os.makedirs(directory, exist_ok=True)

                plt.savefig(f'{directory}OriginalChannel{i + 1}.png')
                plt.close()

        # Calculate Pearson correlation coefficients
        for i in range(49):
            correlation_coefficients_diagonal[i], _ = pearsonr(groundtruth[i], grid_2[i])

        correlation_coefficients_diagonal = correlation_coefficients_diagonal.reshape(7, 7)
        correlation_coefficients_diagonal = np.transpose(correlation_coefficients_diagonal, axes=(1, 0))
        correlation_coefficients_diagonal = np.flipud(correlation_coefficients_diagonal)

        # Plot and save mean_differences_grid
        plt.figure()
        plt.imshow(correlation_coefficients_diagonal, cmap='viridis', origin='upper', vmin=0.85, vmax=1)

        # Set ticks and tick labels before calling imshow
        plt.xticks(range(7), labels=range(1, 8))
        plt.yticks(range(7), labels=range(7, 0, -1))

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
        plt.xticks(np.arange(0, 49, step=5), labels=np.arange(0, 49, step=5))
        plt.yticks(np.arange(0, 49, step=5), labels=np.arange(0, 49, step=5))

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

    file_path = '../testset/Dominik/offline/20230720-091659'
    visualize = Visualizer(file_path)

    # visualize.plot_emg_channels()
    # visualize.plot_emg_channels_difference()
    visualize.plot_correlations()
    # visualize.plot_heat_maps()
    # visualize.plot_pearson_coefficient()
    #visualize.plot_pearson_coefficient_grid()
