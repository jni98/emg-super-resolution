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
        self.emg_data_dicts = self._load_data()
        print("Initialization done")

    def _load_data(self):
        paths = ['path0', 'path1', 'path2', 'path3']
        # Define the specific third dimension sizes for each path
        dim_sizes = {
            'path0': 64576,
            'path1': 60736,
            'path2': 60736,
            'path3': 65984
        }

        data_dicts = []

        for path in paths:
            # Load each file and slice the third dimension as needed
            interpolation_lr_sr = np.load(os.path.join(self.file_path, 'linear_Interpolation', f'{path}-lr.npy'))[:, :,
                                  :dim_sizes[path]]
            cubic_lr_sr = np.load(os.path.join(self.file_path, 'cubic_interpolation', f'{path}-lr.npy'))[:, :,
                          :dim_sizes[path]]
            model1_lr_sr = np.load(os.path.join(self.file_path, 'model_1', f'{path}-lr.npy'))[:, :, :dim_sizes[path]]
            model2_lr_sr = np.load(os.path.join(self.file_path, 'model_2', f'{path}-lr.npy'))[:, :, :dim_sizes[path]]
            model2_Fourier_lr_sr = np.load(os.path.join(self.file_path, 'model_2_Fourier', f'{path}-lr.npy'))[:, :,
                                   :dim_sizes[path]]
            model2_Wavelet_lr_sr = np.load(os.path.join(self.file_path, 'model_2_Wavelet', f'{path}-lr.npy'))[:, :,
                                   :dim_sizes[path]]
            model3_lr_sr = np.load(os.path.join(self.file_path, 'model_3_LSTM', f'{path}-sr.npy')).squeeze()[:, :,
                           :dim_sizes[path]]
            origin_hr = np.load(os.path.join(self.file_path, 'groundtruth', f'{path}-hr.npy'))

            # Create a dictionary with the loaded and adjusted data
            emg_data_dict = {
                path: {  # Associate each dataset with its path
                    'Original': origin_hr,
                    'Bilinear interpolation': interpolation_lr_sr,
                    'Bicubic spline interpolation': cubic_lr_sr,
                    'SRGAN': model1_lr_sr,
                    'UNet': model2_lr_sr,
                    'UNet with Fourier': model2_Fourier_lr_sr,
                    'UNet with Wavelet': model2_Wavelet_lr_sr,
                    'CNN-LSTM': model3_lr_sr
                }
            }

            data_dicts.append(emg_data_dict)

        return data_dicts

    def _create_directory(self, directory: str) -> None:
        """
        Create the directory if it doesn't exist.

        Parameters
        ----------
        directory: str
            Path to the directory to be created.
        """
        os.makedirs(directory, exist_ok=True)

    def plot_emg_channels(self):
        print("Start plotting channels")

        # Iterate through each dictionary in the list self.emg_data_dicts
        for data_dict in self.emg_data_dicts:
            # Each data_dict has only one key-value pair: the path and its associated datasets
            for path, datasets in data_dict.items():
                for title, emg_data in datasets.items():
                    plt.figure(figsize=(8.27, 14))
                    offset = 0
                    for rows in range(13):
                        for cols in range(5):
                            signal = emg_data[rows][cols][
                                     :]  # Adjusted for the third dimension slicing as per your dataset
                            plt.plot(signal.squeeze() + offset, label=f'Channel row {rows + 1} col {cols + 1}')
                            offset += 3500  # Adjust offset for visual clarity

                    plt.xlabel('Time in samples')
                    plt.title(f'{title} - {path}')
                    handles, labels = plt.gca().get_legend_handles_labels()
                    plt.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1))
                    plt.tight_layout()

                    directory = f'plot_channels/{path}/'
                    self._create_directory(directory)
                    plt.savefig(f'{directory}{title}.png')
                    print(f'Saving to: {directory}{title}.png')
                    plt.close()

    def plot_emg_channels_difference(self):
        """
        Plot the mean absolute difference between the channels of the groundtruth signal and other signals.
        """
        print("Start plotting EMG differences")

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original']

                # Iterate over datasets, skipping the 'Original' itself
                for title, emg_data in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    mean_absolute_differences = np.mean(np.abs(groundtruth - emg_data), axis=2)

                    # Plot each channel with a bit of an offset
                    plt.figure(figsize=(12, 8))
                    offset = 0
                    for row in range(13):
                        for col in range(5):
                            diff = np.abs(groundtruth[row, col, :] - emg_data[row, col, :])
                            plt.plot(diff + offset, label=f'Channel row {row + 1} col {col + 1}')
                            offset += np.max(diff) + 1000  # Adjust the offset for visual separation

                    plt.xlabel('Time in samples')
                    plt.ylabel('Mean Absolute Difference (with offset)')
                    plt.title(f'Mean Absolute Differences - Channels - {title} - {path}')
                    plt.legend(loc='upper right', fontsize='small', ncol=2)
                    plt.tight_layout()

                    directory = f'plot_channels_difference/{path}/'
                    self._create_directory(directory)
                    plt.savefig(f'{directory}{title}_channels_difference.png')
                    print(f'Saving to: {directory}{title}_channels_difference.png')
                    plt.close()

                    # Heatmap of mean absolute differences
                    plt.figure()
                    plt.imshow(mean_absolute_differences, cmap='viridis', origin='upper', vmin=0, vmax=60)
                    plt.xticks(range(5), labels=range(1, 6))
                    plt.yticks(range(13), labels=range(13, 0, -1))
                    plt.title(f'Mean Absolute Differences - Heatmap - {title} - {path}')
                    plt.colorbar()

                    directory = f'plot_mean_differences_heatmap/{path}/'
                    self._create_directory(directory)
                    plt.savefig(f'{directory}{title}_mean_differences.png')
                    print(f'Saving to: {directory}{title}_mean_differences.png')
                    plt.close()

    # def plot_heat_maps(self) -> None:
    #     """
    #     Plot heatmaps at specific points in time for each signal.
    #     """
    #     print("Start plotting heat maps")
    #
    #     time_points = [20000, 30000, 40000]
    #     num_signals = len(self.emg_data_dict)
    #     num_time_points = len(time_points)
    #
    #     # Calculate the global min and max for each time point
    #     global_mins = []
    #     global_maxs = []
    #
    #     for i, time_point in enumerate(time_points):
    #         min_val = float('inf')
    #         max_val = float('-inf')
    #         for title, emg_data in self.emg_data_dict.items():
    #             emg_heatmap = emg_data[:, :, time_point]
    #             min_val = min(min_val, np.min(emg_heatmap))
    #             max_val = max(max_val, np.max(emg_heatmap))
    #         global_mins.append(min_val)
    #         global_maxs.append(max_val)
    #
    #     fig, axs = plt.subplots(num_signals, num_time_points, figsize=(15, 30))
    #     for g, (title, emg_data) in enumerate(self.emg_data_dict.items()):
    #         for i, time_point in enumerate(time_points):
    #             emg_heatmap = emg_data[:, :, time_point]
    #             im = axs[g, i].imshow(emg_heatmap, cmap='hot', vmin=global_mins[i], vmax=global_maxs[i])
    #             axs[g, i].set_xticks(range(5))
    #             axs[g, i].set_yticks(range(13))
    #             axs[g, i].set_xticklabels(range(1, 6))
    #             axs[g, i].set_yticklabels(range(1, 14))
    #             axs[g, i].set_title(f'{title} - {time_point}')
    #             axs[g, i].invert_yaxis()
    #             fig.colorbar(im, ax=axs[g, i])
    #
    #     plt.tight_layout()
    #     directory = 'plot_heatmaps/'
    #     self._create_directory(directory)
    #     plt.savefig(f'{directory}Heatmaps.png')
    #     print(f'Saving to: {directory}')
    #     plt.close()
    #
    # def plot_correlations(self) -> None:
    #     """
    #     Plot the cross-correlation of the original signal with other signals.
    #     """
    #     print("Start plotting correlations")
    #
    #     groundtruth = self.emg_data_dict['Original'][:, :, :64570]
    #     for title, emg_data in list(self.emg_data_dict.items())[1:]:
    #         grid_2 = emg_data[:, :, :64570]
    #
    #         number_graphs = 0
    #         plot_number = 1
    #         for row in range(13):
    #             for col in range(5):
    #                 correlation = np.correlate(groundtruth[row, col], grid_2[row, col], mode='full')
    #                 autocorr_groundtruth = np.correlate(groundtruth[row, col], groundtruth[row, col], mode='full')
    #                 normalized_correlation = correlation / np.max(autocorr_groundtruth) if (row != 12 or col != 0) else correlation
    #
    #                 plt.plot(normalized_correlation, label=f'Ch row {row} col {col}')
    #                 number_graphs += 1
    #                 if number_graphs % 5 == 0:
    #                     plt.xlabel('Shift')
    #                     plt.ylabel('Correlation')
    #                     plt.legend(loc='best', bbox_to_anchor=(1, 1))
    #                     plt.title(f'Cross-correlation of {title} with Original HR - {plot_number}')
    #                     plt.ylim(-0.4, 1.2)
    #                     plt.tight_layout()
    #
    #                     directory = f'plot_correlations/{title}/'
    #                     self._create_directory(directory)
    #                     plt.savefig(f'{directory}{title} with Original HR - {plot_number}.png')
    #                     print(f'Saving to: {directory}{title} with Original HR - {plot_number}')
    #                     plt.close()
    #                     plt.figure()
    #                     plot_number += 1

    def plot_normalized_correlations(self):
        """
        Plot the normalized cross-correlation of the original signal with other signals.
        """
        print("Start plotting correlations")

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original']

                auto_corr = np.correlate(groundtruth[0, 0], groundtruth[0, 0], mode='full')
                max_corr = np.max(np.abs(auto_corr))
                max_corr_idx = np.argmax(np.abs(auto_corr))
                print(f"Max autocorrelation: {max_corr} at index {max_corr_idx} for {path}")

                # Iterate over datasets, skipping the 'Original' itself
                for title, emg_data in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    number_graphs = 0
                    plot_number = 1
                    for row in range(13):
                        for col in range(5):
                            x = groundtruth[row, col, :]
                            y = emg_data[row, col, :]
                            mu_x = np.mean(x)
                            mu_y = np.mean(y)
                            std_x = np.std(x)
                            std_y = np.std(y)

                            # Normalize the signals
                            norm_x = (x - mu_x) / std_x
                            norm_y = (y - mu_y) / std_y

                            # Compute normalized cross-correlation
                            correlation = np.correlate(norm_x, norm_y, mode='full')
                            normalized_correlation = correlation / len(
                                x)  # Correct normalization by length of the signal overlap

                            max_corr = np.max(np.abs(normalized_correlation))
                            max_corr_idx = np.argmax(np.abs(normalized_correlation))
                            print(f"Max correlation: {max_corr} at index {max_corr_idx} for {path}")

                            plt.plot(normalized_correlation, label=f'Ch row {row} col {col}')
                            number_graphs += 1
                            if number_graphs % 5 == 0:
                                plt.xlabel('Shift')
                                plt.ylabel('Normalized Correlation')
                                plt.legend(loc='best', bbox_to_anchor=(1, 1))
                                plt.title(
                                    f'Normalized Cross-Correlation of {title} with Original - {plot_number} - {path}')
                                plt.ylim(-1, 1)
                                plt.tight_layout()

                                directory = f'plot_normalized_correlations/{path}/{title}/'
                                self._create_directory(directory)
                                plt.savefig(f'{directory}{title} with Original - {plot_number}.png')
                                print(f'Saving to: {directory}{title} with Original - {plot_number}.png')
                                plt.close()
                                plt.figure()
                                plot_number += 1

    def plot_pearson_coefficient_grid(self):
        """
        Plot the Pearson coefficient of the correlation of one channel in the groundtruth signal with all other channels.
        """
        print("Start plotting Pearson coefficient grids")

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original']

                # Iterate over datasets, skipping the 'Original' itself
                for title, grid_2 in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    correlation_coefficients = np.zeros((13, 5))

                    for row in range(13):
                        for col in range(5):
                            correlation_coefficients[row, col], _ = pearsonr(groundtruth[row, col], grid_2[row, col])

                    plt.figure()
                    plt.imshow(correlation_coefficients, cmap='viridis', origin='upper', vmin=0.95, vmax=1)
                    plt.xticks(range(5), labels=range(1, 6))
                    plt.yticks(range(13), labels=range(13, 0, -1))
                    plt.title(f'Pearson coefficient - Origin with {title} - {path}')
                    plt.colorbar()

                    directory = f'plot_pearsonr_heatmap/{path}/'
                    self._create_directory(directory)
                    plt.savefig(f'{directory}{title}.png')
                    print(f'Saving to: {directory}{title}.png')
                    plt.close()

                    for row1 in range(13):
                        for col1 in range(5):
                            for row2 in range(13):
                                for col2 in range(5):
                                    correlation_coefficients[row2, col2], _ = pearsonr(groundtruth[row1, col1],
                                                                                       grid_2[row2, col2])

                            plt.figure()
                            plt.imshow(correlation_coefficients, cmap='viridis', origin='upper', vmin=0.3, vmax=1)
                            plt.xticks(range(5), labels=range(1, 6))
                            plt.yticks(range(13), labels=range(13, 0, -1))
                            plt.title(f'Pearson coefficient - Origin channel ({row1, col1}) with {title} - {path}')
                            plt.colorbar()

                            directory = f'plot_pearsonr_heatmap_oneChannel/{path}/'
                            self._create_directory(directory)
                            plt.savefig(f'{directory}{title}-r{row1}c{col1}.png')
                            print(f'Saving to: {directory}{title}-r{row1}c{col1}.png')
                            plt.close()

    def create_difference_plots(self):
        """
        Plot the absolute differences between the groundtruth signal and other signals.
        """
        print("Start plotting absolute difference grids")

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original']

                # Iterate over datasets, skipping the 'Original' itself
                for title, grid_2 in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    diff = groundtruth - grid_2
                    global_min = np.floor(np.min(diff) / 200) * 200
                    global_max = np.ceil(np.max(diff) / 200) * 200

                    # Main figure with subplots
                    fig, axes = plt.subplots(nrows=13, ncols=5, figsize=(20, 20))
                    fig.suptitle(f'Differences: Original vs {title} - {path}', fontsize=16)

                    # Additional figure for the large plot at position [1][4]
                    fig_large, ax_large = plt.subplots(figsize=(50, 8.27))  # A4 page in inches

                    for i in range(13):
                        for j in range(5):
                            ax = axes[i, j]
                            ax.plot(diff[i, j])
                            ax.set_title(f'Row {i + 1}, Column {j + 1}', fontsize=8)
                            ax.set_xticks([])
                            if j == 0:
                                y_ticks = np.arange(global_min, global_max + 200, 200)
                                ax.set_yticks(y_ticks)
                                ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])
                            else:
                                ax.set_yticks([])
                            ax.set_ylim(global_min, global_max)
                            if i == 1 and j == 4:
                                ax_large.plot(diff[i, j])
                                ax_large.set_title(f'Large Plot: Differences at Row 2, Column 5 - {path}', fontsize=12)
                                ax_large.set_xlabel('Time Points')
                                ax_large.set_ylabel('Difference Value')
                                ax_large.set_ylim(global_min, global_max)
                                y_ticks_large = np.arange(global_min, global_max + 200, 200)
                                ax_large.set_yticks(y_ticks_large)
                                ax_large.set_yticklabels([f'{tick:.0f}' for tick in y_ticks_large])

                    fig.tight_layout()
                    fig.subplots_adjust(top=0.95)
                    directory = f'plot_difference/{path}/'
                    self._create_directory(directory)
                    fig.savefig(f'{directory}Differences_{title}.png')
                    print(f'Saving to: {directory}Differences_{title}.png')
                    plt.close(fig)
                    fig_large.tight_layout()
                    fig_large.savefig(f'{directory}Differences_Large_Plot_Row_2_Column_5_{title}.png')
                    print(f'Saving large plot to: {directory}Differences_Large_Plot_Row_2_Column_5_{title}.png')
                    plt.close(fig_large)

        print("Finished plotting all differences.")

    def propagation_plot(self) -> None:
        """
        Creates a figure with 13 subplots for each element in the grids dictionary.
        Each subplot shows the signal [y][0][28976:31024] for the first column of the array.
        """
        print("Start plotting propagation plots")

        start_idx = 30050
        end_idx = start_idx + 256

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                # Iterate over datasets
                for title, array in datasets.items():
                    fig, axes = plt.subplots(nrows=13, ncols=1, figsize=(15, 30))
                    fig.suptitle(f'Propagation Plot: {title} - {path}', fontsize=16)

                    for y in range(13):
                        ax = axes[y]
                        ax.plot(array[y, 2, start_idx:end_idx])  # Adjusted to plot from the third dimension
                        ax.set_title(f'Row {y + 1}', fontsize=10)
                        ax.set_xticks([])
                        ax.set_yticks([])

                    plt.tight_layout()
                    plt.subplots_adjust(top=0.95)
                    directory = f'plot_propagation/{path}/'
                    self._create_directory(directory)
                    plt.savefig(f'{directory}Propagation_Plot_{title}.png')
                    print(f'Saving to: {directory}Propagation_Plot_{title}.png')
                    plt.close()

        print("Finished plotting propagation plots.")

    def calculate_windowed_correlation(self, signal1, signal2, window_size=512):
        """
        Calculate the Pearson correlation coefficient for non-overlapping windows of two signals.

        Parameters
        ----------
        signal1 : array-like
            First signal.
        signal2 : array-like
            Second signal.
        window_size : int
            Size of the windows.

        Returns
        -------
        list
            List of Pearson correlation coefficients for each window.
        """

        signal1 = signal1.astype(np.float64)
        signal2 = signal2.astype(np.float64)

        num_windows = len(signal1) // window_size
        correlations = []
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window1 = signal1[start:end]
            window2 = signal2[start:end]
            if len(window1) == len(window2):  # Ensure both windows are of the same size
                corr, _ = pearsonr(window1, window2)
                if corr == 0.9999999999999999:
                    corr = 1

                if corr < 0:
                    print("Negative correlation")
                correlations.append(corr)
        return correlations

    def plot_windowed_correlations(self):
        """
        Plot the Pearson correlation coefficient for non-overlapping windows of the original signal with other signals,
        and save all single plots as well as one big figure with 13x5 subplots for each signal.
        """
        print("Start plotting windowed correlations")
        window_size = 512

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original']
                linear_interpolation = datasets['Bilinear interpolation']

                # Iterate over datasets, skipping the 'Original' itself
                for title, emg_data in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    directory = f'plot_windowed_correlations/{path}/{title}/'
                    self._create_directory(directory)

                    # Setup the figure for the big plot (13x5 grid of subplots)
                    fig, axes = plt.subplots(nrows=13, ncols=5, figsize=(20, 20))

                    for row in range(13):
                        for col in range(5):
                            groundtruth_sig = groundtruth[row, col, :]
                            signal2 = emg_data[row, col, :]
                            correlations = self.calculate_windowed_correlation(groundtruth_sig, signal2, window_size)
                            x_axis = np.arange(len(correlations))

                            # Plot and save individual correlation plots
                            plt.figure()
                            plt.plot(x_axis, correlations)
                            plt.xlabel('Window Index')
                            plt.ylabel('Pearson Correlation Coefficient')
                            plt.title(
                                f'Windowed Pearson Correlation - Electrode ({row + 1}, {col + 1}) - Original with {title}')
                            plt.ylim(0, 1.1)
                            plt.savefig(f'{directory}Windowed_Correlation_Electrode_{row + 1}_{col + 1}.png')
                            plt.close()

                            # Add to big subplot figure
                            ax = axes[row, col]
                            ax.plot(x_axis, correlations)
                            ax.set_title(f'Electrode ({row + 1}, {col + 1})')
                            ax.set_ylim(0, 1.1)
                            ax.label_outer()  # Only show outer labels to reduce clutter

                    # Save the large figure with all subplots
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.95)
                    fig.suptitle(f'Combined Windowed Pearson Correlation - {title}')
                    fig.savefig(f'{directory}Combined_Windowed_Correlation_{title}.png')
                    print(f'Saving to: {directory}Combined_Windowed_Correlation_{title}.png')

                    plt.close(fig)

                    # Plot differences if the current title is not 'Bilinear interpolation'
                    if title != 'Bilinear interpolation':
                        diff_directory = f'plot_windowed_correlations/{path}/Linear_vs_{title}/'
                        self._create_directory(diff_directory)
                        fig_diff, axes_diff = plt.subplots(nrows=13, ncols=5, figsize=(20, 20))

                        for row in range(13):
                            for col in range(5):
                                signal_linear = linear_interpolation[row, col, :]
                                signal_compare = emg_data[row, col, :]
                                correlations_linear = self.calculate_windowed_correlation(groundtruth[row, col, :],
                                                                                          signal_linear, window_size)
                                correlations_compare = self.calculate_windowed_correlation(groundtruth[row, col, :],
                                                                                           signal_compare, window_size)
                                differences = np.array(correlations_compare) - np.array(correlations_linear)
                                x_axis_diff = np.arange(len(differences))

                                # Individual plot for differences
                                plt.figure()
                                plt.fill_between(x_axis_diff, differences, where=(differences >= 0), color='lightgreen',
                                                 alpha=0.5)
                                plt.fill_between(x_axis_diff, differences, where=(differences < 0), color='lightcoral',
                                                 alpha=0.5)
                                plt.plot(x_axis_diff, differences, color='black')
                                plt.xlabel('Window Index')
                                plt.ylabel('Difference in Pearson Correlation Coefficient')
                                plt.title(
                                    f'Difference in Windowed Pearson Correlation - Electrode ({row + 1}, {col + 1}) - Linear vs {title}')
                                plt.ylim(-0.15, 0.15)
                                plt.savefig(
                                    f'{diff_directory}Difference_Windowed_Correlation_Electrode_{row + 1}_{col + 1}.png')
                                plt.close()

                                # Add to big subplot figure for differences
                                ax_diff = axes_diff[row, col]
                                ax_diff.fill_between(x_axis_diff, differences, where=(differences >= 0),
                                                     color='lightgreen', alpha=0.5)
                                ax_diff.fill_between(x_axis_diff, differences, where=(differences < 0),
                                                     color='lightcoral', alpha=0.5)
                                ax_diff.plot(x_axis_diff, differences, color='black')
                                ax_diff.set_ylim(-0.15, 0.15)
                                ax_diff.label_outer()

                        # Save the large difference figure
                        fig_diff.tight_layout()
                        fig_diff.subplots_adjust(top=0.95)
                        fig_diff.suptitle(f'Combined Difference in Windowed Pearson Correlation - Linear vs {title}')
                        fig_diff.savefig(f'{diff_directory}Combined_Difference_{title}.png')
                        print(f'Saving to: {diff_directory}Combined_Difference_{title}.png')

                        plt.close(fig_diff)

        print("Finished plotting all correlations and differences.")

    def create_box_plot(self):
        data_dicts = self._load_data()

        # Initialize dictionaries to hold differences for each model across all paths for each subset
        differences_all = {
            'Bilinear interpolation': [],
            'Bicubic spline interpolation': [],
            'SRGAN': [],
            'UNet': [],
            'UNet with Fourier': [],
            'UNet with Wavelet': [],
            'CNN-LSTM': []
        }

        differences_even = {
            'Bilinear interpolation': [],
            'Bicubic spline interpolation': [],
            'SRGAN': [],
            'UNet': [],
            'UNet with Fourier': [],
            'UNet with Wavelet': [],
            'CNN-LSTM': []
        }

        differences_odd = {
            'Bilinear interpolation': [],
            'Bicubic spline interpolation': [],
            'SRGAN': [],
            'UNet': [],
            'UNet with Fourier': [],
            'UNet with Wavelet': [],
            'CNN-LSTM': []
        }

        # Calculate the absolute differences
        for data_dict in data_dicts:
            for path, data in data_dict.items():
                origin_hr = data['Original']
                for model, model_data in data.items():
                    if model != 'Original':
                        # All data
                        differences_all[model].extend(np.abs(model_data - origin_hr).flatten())

                        # Even rows and columns (0,0), (0,2), (2,0), etc.
                        differences_even[model].extend(np.abs(model_data[::2, ::2] - origin_hr[::2, ::2]).flatten())

                        # Odd rows and columns (1,1), (1,3), (3,1), etc.
                        differences_odd[model].extend(np.abs(model_data[1::2, 1::2] - origin_hr[1::2, 1::2]).flatten())

        def calculate_outliers(data):
            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)

            # Calculate IQR
            IQR = Q3 - Q1

            # Calculate lower and upper bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Identify outliers
            outliers = [x for x in data if x < lower_bound or x > upper_bound]

            # Filter outliers from the data
            filtered_data = [x for x in data if lower_bound <= x <= upper_bound]

            # Calculate the percentage of outliers
            outliers_percentage = (len(outliers) / len(data)) * 100

            return outliers_percentage, filtered_data

        def plot_boxplot(differences, title, colors, filename, remove_outliers=False, show_percentage=True):
            boxplot_data = []
            labels = []
            outlier_percentages = []

            for model in differences:
                if remove_outliers:
                    outlier_percentage, filtered_data = calculate_outliers(differences[model])
                    boxplot_data.append(filtered_data)
                    outlier_percentages.append(outlier_percentage)
                else:
                    boxplot_data.append(differences[model])
                    outlier_percentage, _ = calculate_outliers(differences[model])
                    outlier_percentages.append(outlier_percentage)
                labels.append(model)

            fig, ax = plt.subplots(figsize=(18, 10))
            bp = ax.boxplot(boxplot_data, labels=labels, patch_artist=True)

            for patch, color in zip(bp['boxes'], colors * (len(differences) // len(colors))):
                patch.set_facecolor(color)

            if show_percentage:
                for i, outlier_percentage in enumerate(outlier_percentages):
                    # Move the percentage text upwards by multiplying the maximum value by 1.025
                    max_val = max(boxplot_data[i]) if boxplot_data[i] else 0  # Handle empty lists
                    ax.text(i + 1, max_val * 1.025, f'{outlier_percentage:.2f}%', horizontalalignment='center',
                            color='red')

            plt.title(title)
            plt.xlabel('Models')
            plt.ylabel('Absolute Difference')
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(filename)
            plt.close()

        colors = ['#C6DEE9', '#5DC08A', '#2F586E']

        # Plot for all data
        plot_boxplot(differences_all, 'Absolute Differences Between Models and Ground Truth (All Data)', colors,
                     'boxplots/boxplot_all_data.png')
        plot_boxplot(differences_all,
                     'Absolute Differences Between Models and Ground Truth (All Data) - Without Outliers', colors,
                     'boxplots/boxplot_all_data_no_outliers.png', remove_outliers=True, show_percentage=False)

        # Plot for even rows and columns
        plot_boxplot(differences_even, 'Absolute Differences Between Models and Ground Truth (Even Rows/Cols)', colors,
                     'boxplots/boxplot_even_rows_cols.png')
        plot_boxplot(differences_even,
                     'Absolute Differences Between Models and Ground Truth (Even Rows/Cols) - Without Outliers', colors,
                     'boxplots/boxplot_even_rows_cols_no_outliers.png', remove_outliers=True, show_percentage=False)

        # Plot for odd rows and columns
        plot_boxplot(differences_odd, 'Absolute Differences Between Models and Ground Truth (Odd Rows/Cols)', colors,
                     'boxplots/boxplot_odd_rows_cols.png')
        plot_boxplot(differences_odd,
                     'Absolute Differences Between Models and Ground Truth (Odd Rows/Cols) - Without Outliers', colors,
                     'boxplots/boxplot_odd_rows_cols_no_outliers.png', remove_outliers=True, show_percentage=False)

    def linear_methods_subtraction(self):

        # subtract linear interpolation and linear zoom interpolation
        groundtruth = self.emg_data_dict['Original']
        linear_interpolation = self.emg_data_dict['Linear interpolation']
        linear_zoom_interpolation = self.emg_data_dict['Linear Zoom LR to SR']

        # cut lkength
        groundtruth = groundtruth[:, :, :]
        linear_interpolation = linear_interpolation[:, :, :]
        linear_zoom_interpolation = linear_zoom_interpolation[:, :, :]

        difference = linear_interpolation - linear_zoom_interpolation

        # print all indices that are unequal zero
        # for row in range(13):
        #     for col in range(5):
        #         if np.any(difference[row, col] != 0):
        #             print(f"Row {row} Col {col}, {difference[row, col]}")

        # check which of the linear models is closer to the groundtruth
        for row in range(13):
            for col in range(5):
                diff_linear = np.sum(np.abs(groundtruth[row, col] - linear_interpolation[row, col]))
                diff_zoom = np.sum(np.abs(groundtruth[row, col] - linear_zoom_interpolation[row, col]))
                print(f"Row {row} Col {col}: Linear {diff_linear}, Zoom {diff_zoom}")


if __name__ == '__main__':
    file_path = '../results/'
    visualize = Visualizer(file_path)
    #visualize.plot_emg_channels()
    #visualize.plot_emg_channels_difference()

    #visualize.plot_normalized_correlations()
    #visualize.plot_pearson_coefficient_grid()
    #visualize.create_difference_plots()
    #visualize.propagation_plot()
    #visualize.plot_windowed_correlations()

    visualize.create_box_plot()