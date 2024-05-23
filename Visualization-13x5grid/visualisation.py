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
        self.emg_data_dict = self._load_data()
        print("Initialization done")

    def _load_data(self) -> dict:
        """
        Load the data from the given file path.

        Returns
        -------
        dict
            Dictionary with titles as keys and EMG data as values.
        """
        interpolation_lr_sr = np.load(self.file_path + 'linear_Interpolation/path0-lr.npy')
        model1_lr_sr = np.load(self.file_path + 'model_1/path0-lr.npy')
        model2_lr_sr = np.load(self.file_path + 'model_2/path0-lr.npy')
        model2_Fourier_lr_sr = np.load(self.file_path + 'model_2_Fourier/path0-lr.npy')
        model2_Wavelet_lr_sr = np.load(self.file_path + 'model_2_Wavelet/path0-lr.npy')
        model3_lr_sr = np.load(self.file_path + 'model_3_LSTM/path0-sr-b.npy').squeeze()
        origin_hr = np.load(self.file_path + 'groundtruth/path0-hr.npy')

        return {
            'Original HR': origin_hr,
            'Linear LR to SR': interpolation_lr_sr,
            'Model 1 LR to SR': model1_lr_sr,
            'Model 2 LR to SR': model2_lr_sr,
            'Model 2 Fourier LR to SR': model2_Fourier_lr_sr,
            'Model 2 Wavelet LR to SR': model2_Wavelet_lr_sr,
            'Model 3 LR to SR': model3_lr_sr
        }

    def _create_directory(self, directory: str) -> None:
        """
        Create the directory if it doesn't exist.

        Parameters
        ----------
        directory: str
            Path to the directory to be created.
        """
        os.makedirs(directory, exist_ok=True)

    def plot_emg_channels(self) -> None:
        """
        Plot all EMG channels for the original signal, the model 2 signal, and the interpolated signal.
        Channels are plotted with an offset.
        """
        print("Start plotting channels")

        time_axis = np.arange(64570)
        for title, emg_data in self.emg_data_dict.items():
            plt.figure(figsize=(8.27, 14))
            offset = 0
            for rows in range(13):
                for cols in range(5):
                    signal = emg_data[rows][cols][:64570]
                    plt.plot(time_axis, signal.squeeze() + offset, label=f'Channel row{rows + 1} col{cols + 1}')
                    offset += 3500

            plt.xlabel('Time in samples')
            plt.title(title)
            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1))
            plt.tight_layout()

            directory = 'plot_channels/'
            self._create_directory(directory)
            plt.savefig(f'{directory}{title}.png')
            print(f'Saving to: {directory}{title}')
            plt.close()

    def plot_emg_channels_difference(self) -> None:
        """
        Plot the mean absolute difference between the channels of the groundtruth signal and other signals.
        """
        print("Start plotting EMG differences")

        groundtruth = self.emg_data_dict['Original HR'][:, :, :64570]

        for title, emg_data in list(self.emg_data_dict.items())[1:]:  # Skip the groundtruth itself
            emg_data_cut = emg_data[:, :, :64570]
            mean_absolute_differences = np.mean(np.abs(groundtruth - emg_data_cut), axis=2)

            # Plot each channel with a bit of an offset
            plt.figure(figsize=(12, 8))
            offset = 0
            for row in range(13):
                for col in range(5):
                    diff = np.abs(groundtruth[row, col, :] - emg_data_cut[row, col, :])
                    plt.plot(diff + offset, label=f'Channel row {row + 1} col {col + 1}')
                    offset += np.max(diff) + 1000  # Adjust the offset for visual separation

            plt.xlabel('Time in samples')
            plt.ylabel('Mean Absolute Difference (with offset)')
            plt.title(f'Mean Absolute Differences - Channels - {title}')
            plt.legend(loc='upper right', fontsize='small', ncol=2)
            plt.tight_layout()

            directory = 'plot_channels_difference/'
            self._create_directory(directory)
            plt.savefig(f'{directory}{title}_channels_difference.png')
            print(f'Saving to: {directory}{title}_channels_difference.png')
            plt.close()

            # Heatmap of mean absolute differences
            plt.figure()
            plt.imshow(mean_absolute_differences, cmap='viridis', origin='upper', vmin=0, vmax=60)
            plt.xticks(range(5), labels=range(1, 6))
            plt.yticks(range(13), labels=range(13, 0, -1))
            plt.title(f'Mean Absolute Differences - Heatmap - {title}')
            plt.colorbar()

            directory = 'plot_mean_differences_heatmap/'
            self._create_directory(directory)
            plt.savefig(f'{directory}{title}_mean_differences.png')
            print(f'Saving to: {directory}{title}_mean_differences.png')
            plt.close()

    def plot_heat_maps(self) -> None:
        """
        Plot heatmaps at specific points in time for each signal.
        """
        print("Start plotting heat maps")

        time_points = [20000, 30000, 40000]
        num_signals = len(self.emg_data_dict)
        num_time_points = len(time_points)

        # Calculate the global min and max for each time point
        global_mins = []
        global_maxs = []

        for i, time_point in enumerate(time_points):
            min_val = float('inf')
            max_val = float('-inf')
            for title, emg_data in self.emg_data_dict.items():
                emg_heatmap = emg_data[:, :, time_point]
                min_val = min(min_val, np.min(emg_heatmap))
                max_val = max(max_val, np.max(emg_heatmap))
            global_mins.append(min_val)
            global_maxs.append(max_val)

        fig, axs = plt.subplots(num_signals, num_time_points, figsize=(15, 30))
        for g, (title, emg_data) in enumerate(self.emg_data_dict.items()):
            for i, time_point in enumerate(time_points):
                emg_heatmap = emg_data[:, :, time_point]
                im = axs[g, i].imshow(emg_heatmap, cmap='hot', vmin=global_mins[i], vmax=global_maxs[i])
                axs[g, i].set_xticks(range(5))
                axs[g, i].set_yticks(range(13))
                axs[g, i].set_xticklabels(range(1, 6))
                axs[g, i].set_yticklabels(range(1, 14))
                axs[g, i].set_title(f'{title} - {time_point}')
                axs[g, i].invert_yaxis()
                fig.colorbar(im, ax=axs[g, i])

        plt.tight_layout()
        directory = 'plot_heatmaps/'
        self._create_directory(directory)
        plt.savefig(f'{directory}Heatmaps.png')
        print(f'Saving to: {directory}')
        plt.close()

    def plot_correlations(self) -> None:
        """
        Plot the cross-correlation of the original signal with other signals.
        """
        print("Start plotting correlations")

        groundtruth = self.emg_data_dict['Original HR'][:, :, :64570]
        for title, emg_data in list(self.emg_data_dict.items())[1:]:
            grid_2 = emg_data[:, :, :64570]

            number_graphs = 0
            plot_number = 1
            for row in range(13):
                for col in range(5):
                    correlation = np.correlate(groundtruth[row, col], grid_2[row, col], mode='full')
                    autocorr_groundtruth = np.correlate(groundtruth[row, col], groundtruth[row, col], mode='full')
                    normalized_correlation = correlation / np.max(autocorr_groundtruth) if (row != 12 or col != 0) else correlation

                    plt.plot(normalized_correlation, label=f'Ch row {row} col {col}')
                    number_graphs += 1
                    if number_graphs % 5 == 0:
                        plt.xlabel('Shift')
                        plt.ylabel('Correlation')
                        plt.legend(loc='best', bbox_to_anchor=(1, 1))
                        plt.title(f'Cross-correlation of {title} with Original HR - {plot_number}')
                        plt.ylim(-0.4, 1.2)
                        plt.tight_layout()

                        directory = f'plot_correlations/{title}/'
                        self._create_directory(directory)
                        plt.savefig(f'{directory}{title} with Original HR - {plot_number}.png')
                        print(f'Saving to: {directory}{title} with Original HR - {plot_number}')
                        plt.close()
                        plt.figure()
                        plot_number += 1

    def plot_pearson_coefficient_grid(self) -> None:
        """
        Plot the Pearson coefficient of the correlation of one channel in the groundtruth signal with all other channels.
        """
        print("Start plotting Pearson coefficient grids")

        groundtruth = self.emg_data_dict['Original HR'][:, :, :64570]
        for title, grid_2 in list(self.emg_data_dict.items())[1:]:
            grid_2 = grid_2[:, :, :64570]
            correlation_coefficients = np.zeros((13, 5))

            for row in range(13):
                for col in range(5):
                    correlation_coefficients[row, col], _ = pearsonr(groundtruth[row, col], grid_2[row, col])

            plt.figure()
            plt.imshow(correlation_coefficients, cmap='viridis', origin='upper', vmin=0.95, vmax=1)
            plt.xticks(range(5), labels=range(1, 6))
            plt.yticks(range(13), labels=range(13, 0, -1))
            plt.title(f'Pearson coefficient - Origin with {title}')
            plt.colorbar()

            directory = 'plot_pearsonr_heatmap/'
            self._create_directory(directory)
            plt.savefig(f'{directory}{title}.png')
            print(f'Saving to: {directory}{title}')
            plt.close()

            for row1 in range(13):
                for col1 in range(5):
                    for row2 in range(13):
                        for col2 in range(5):
                            correlation_coefficients[row2, col2], _ = pearsonr(groundtruth[row1, col1], grid_2[row2, col2])

                    plt.figure()
                    plt.imshow(correlation_coefficients, cmap='viridis', origin='upper', vmin=0.3, vmax=1)
                    plt.xticks(range(5), labels=range(1, 6))
                    plt.yticks(range(13), labels=range(13, 0, -1))
                    plt.title(f'Pearson coefficient - Origin channel ({row1, col1}) with {title}')
                    plt.colorbar()

                    directory = 'plot_pearsonr_heatmap_oneChannel/'
                    self._create_directory(directory)
                    plt.savefig(f'{directory}{title}-r{row1}c{col1}.png')
                    print(f'Saving to: {directory}{title}-r{row1}c{col1}')
                    plt.close()

    def create_difference_plots(self) -> None:
        """
        Creates subplots showing the absolute difference between the original data and other models.
        """
        print("Start plotting absolute difference grids")

        groundtruth = self.emg_data_dict['Original HR'][:, :, :64570]
        for title, grid_2 in list(self.emg_data_dict.items())[1:]:
            grid_2 = grid_2[:, :, :64570]
            abs_diff = np.abs(groundtruth - grid_2)
            global_min = np.floor(np.min(abs_diff) / 200) * 200
            global_max = np.ceil(np.max(abs_diff) / 200) * 200

            fig, axes = plt.subplots(nrows=13, ncols=5, figsize=(20, 20))
            fig.suptitle(f'Absolute Differences: Original HR vs {title}', fontsize=16)
            for i in range(13):
                for j in range(5):
                    ax = axes[i, j]
                    ax.plot(abs_diff[i, j])
                    ax.set_title(f'Row {i + 1}, Column {j + 1}', fontsize=8)
                    ax.set_xticks([])

                    if j == 0:
                        y_ticks = np.arange(global_min, global_max + 200, 200)
                        ax.set_yticks(y_ticks)
                        ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])
                    else:
                        ax.set_yticks([])

                    ax.set_ylim(global_min, global_max)

            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            directory = 'plot_absolute_difference/'
            self._create_directory(directory)
            plt.savefig(f'{directory}Absolute_Differences_{title}.png')
            print(f'Saving to: {directory}Absolute_Differences_{title}')
            plt.close()

    def propagation_plot(self) -> None:
        """
        Creates a figure with 13 subplots for each element in the grids dictionary.
        Each subplot shows the signal [y][0][28976:31024] for the first column of the array.
        """
        print("Start plotting propagation plots")

        start_idx = 30050
        end_idx = start_idx + 256

        for title, array in self.emg_data_dict.items():
            fig, axes = plt.subplots(nrows=13, ncols=1, figsize=(15, 30))
            fig.suptitle(f'Propagation Plot: {title}', fontsize=16)
            for y in range(13):
                ax = axes[y]
                ax.plot(array[y, 2, start_idx:end_idx])
                ax.set_title(f'Row {y + 1}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            directory = 'plot_propagation/'
            self._create_directory(directory)
            plt.savefig(f'{directory}Propagation_Plot_{title}.png')
            print(f'Saving to: {directory}Propagation_Plot_{title}')
            plt.close()

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

    def plot_windowed_correlations(self) -> None:
        """
        Plot the Pearson correlation coefficient for non-overlapping windows of the original signal with other signals.
        """
        print("Start plotting windowed correlations")

        window_size = 512
        groundtruth = self.emg_data_dict['Original HR']
        linear_interpolation = self.emg_data_dict['Linear LR to SR']

        for title, emg_data in list(self.emg_data_dict.items())[1:]:
            # Plot correlations for each model
            for row in range(13):
                for col in range(5):
                    groundtruth_sig= groundtruth[row, col, :64570]
                    signal2 = emg_data[row, col, :64570]
                    correlations = self.calculate_windowed_correlation(groundtruth_sig, signal2, window_size)
                    x_axis = np.arange(len(correlations))

                    plt.figure()
                    plt.plot(x_axis, correlations)
                    plt.xlabel('Window Index')
                    plt.ylabel('Pearson Correlation Coefficient')
                    plt.title(f'Windowed Pearson Correlation - Electrode ({row + 1}, {col + 1}) - Original HR with {title}')
                    plt.ylim(0, 1.1)

                    directory = f'plot_windowed_correlations/{title}/'
                    self._create_directory(directory)
                    plt.savefig(f'{directory}Windowed_Correlation_Electrode_{row + 1}_{col + 1}.png')
                    print(f'Saving to: {directory}Windowed_Correlation_Electrode_{row + 1}_{col + 1}.png')
                    plt.close()

            # Plot differences with the linear interpolation model
            if title != 'Linear LR to SR':
                for row in range(13):
                    for col in range(5):
                        signal_linear = linear_interpolation[row, col, :64570]
                        signal_compare = emg_data[row, col, :64570]
                        groundtruth_sig = groundtruth[row, col, :64570]

                        if title == 'Model 1 LR to SR':
                            print("STOP")
                        correlations_linear = self.calculate_windowed_correlation(groundtruth_sig, signal_linear, window_size)
                        correlations_compare = self.calculate_windowed_correlation(groundtruth_sig, signal_compare, window_size)

                        differences = np.array(correlations_compare) - np.array(correlations_linear)
                        x_axis = np.arange(len(differences))

                        plt.figure()
                        plt.fill_between(x_axis, differences, where=(differences >= 0),color='lightgreen', alpha=0.5)
                        plt.fill_between(x_axis, differences, where=(differences < 0), color='lightcoral', alpha=0.5)
                        plt.plot(x_axis, differences, color='black')
                        plt.xlabel('Window Index')
                        plt.ylabel('Difference in Pearson Correlation Coefficient')
                        plt.title(f'Difference in Windowed Pearson Correlation - Electrode ({row + 1}, {col + 1}) - Linear vs {title}')
                        plt.ylim(-0.15, 0.15)

                        directory = f'plot_windowed_correlations/Linear_vs_{title}/'
                        self._create_directory(directory)
                        plt.savefig(f'{directory}Difference_Windowed_Correlation_Electrode_{row + 1}_{col + 1}.png')
                        print(f'Saving to: {directory}Difference_Windowed_Correlation_Electrode_{row + 1}_{col + 1}.png')
                        plt.close()



if __name__ == '__main__':
    file_path = '../results/'
    visualize = Visualizer(file_path)
    # visualize.plot_emg_channels()
    #visualize.plot_emg_channels_difference()
    # visualize.plot_heat_maps()
    # visualize.plot_correlations()
    # visualize.plot_pearson_coefficient_grid()
    #visualize.create_difference_plots()
    #visualize.propagation_plot()
    visualize.plot_windowed_correlations()