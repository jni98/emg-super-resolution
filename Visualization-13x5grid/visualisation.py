"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Visualizer, which is used to visualize the results of the interpolation
and the neural networks.

The code is started by calling the main function that creates an object of the Visualizer class. The object is then used
to call the different visualization functions.
"""

import os
import textwrap

import numpy as np
import matplotlib
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
import json


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
        plt.rcParams.update({'font.size': 10})
        self.file_path = file_path
        self.emg_data_dicts = self._load_data()
        print("Initialization done")

    def _load_data(self):
        #paths = ['path0', 'path1', 'path2', 'path3']
        paths = ['path0']
        # Define the specific third dimension sizes for each path
        dim_sizes = {
            'path0': 64576,
            #'path1': 60736,
            #'path2': 60736,
            #'path3': 65984
        }

        data_dicts = []

        def butter_bandpass(lowcut, highcut, fs, order=2):
            nyquist = 0.5 * fs
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def apply_bandpass_filter(data, lowcut=20, highcut=500, fs=2048, order=2):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            filtered_data = filtfilt(b, a, data, axis=2)
            return filtered_data

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

            # Apply bandpass filter
            interpolation_lr_sr = apply_bandpass_filter(interpolation_lr_sr)
            cubic_lr_sr = apply_bandpass_filter(cubic_lr_sr)
            model1_lr_sr = apply_bandpass_filter(model1_lr_sr)
            model2_lr_sr = apply_bandpass_filter(model2_lr_sr)
            model2_Fourier_lr_sr = apply_bandpass_filter(model2_Fourier_lr_sr)
            model2_Wavelet_lr_sr = apply_bandpass_filter(model2_Wavelet_lr_sr)
            model3_lr_sr = apply_bandpass_filter(model3_lr_sr)
            origin_hr = apply_bandpass_filter(origin_hr)

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
        directory = directory.replace(' ', '_')
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
                    file_name = f'{directory}{title}.pdf'
                    file_name = file_name.replace(' ', '_')
                    plt.savefig(file_name,format='pdf')
                    print(f'Saving to: {directory}{title}.pdf')
                    plt.close()

    def plot_emg_channels_difference(self):
        """
        Plot the mean absolute difference between the channels of the groundtruth signal and other signals.
        """
        print("Start plotting EMG differences")

        for data_dict in self.emg_data_dicts:
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original']
                model_titles = [title for title in datasets.keys() if title != 'Original']
                mean_diffs_dict = {}

                for title, emg_data in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    mean_absolute_differences = np.mean(np.abs(groundtruth - emg_data), axis=2)
                    mean_diffs_dict[title] = mean_absolute_differences

                    plt.figure()
                    offset = 0
                    for row in range(13):
                        for col in range(5):
                            diff = np.abs(groundtruth[row, col, :] - emg_data[row, col, :])
                            plt.plot(diff + offset, label=f'Channel row {row + 1} col {col + 1}')
                            offset += np.max(diff) + 1000

                    plt.xlabel('Time in samples')
                    plt.ylabel('Mean Absolute Difference (with offset)')
                    plt.title(f'{title}')
                    plt.legend(loc='upper right', fontsize='small', ncol=2)
                    plt.tight_layout(rect=[0, 0, 1, 0.95])

                    directory = f'plot_channels_difference/{path}/'
                    self._create_directory(directory)
                    filename = f'{directory}{title}_channels_difference.pdf'
                    filename = filename.replace(' ', '_')
                    plt.savefig(filename,format='pdf')
                    print(f'Saving to: {directory}{title}_channels_difference.pdf')
                    plt.close()

                    plt.figure()
                    plt.imshow(mean_absolute_differences, cmap='viridis', origin='upper', vmin=0, vmax=60)
                    plt.xticks(range(5), labels=range(1, 6))
                    plt.yticks(range(13), labels=range(13, 0, -1))
                    plt.title(f'Mean Absolute Differences - {title}')
                    plt.colorbar()

                    directory = f'plot_mean_differences_heatmap/{path}/'
                    self._create_directory(directory)
                    filename = f'{directory}{title}_mean_differences.pdf'
                    filename = filename.replace(' ', '_')
                    plt.savefig(filename, bbox_inches='tight',format='pdf')
                    print(f'Saving to: {directory}{title}_mean_differences.pdf')
                    plt.close()

                fig_width_cm = 16  # New width
                fig_height_cm = fig_width_cm * 0.9  # New height
                fig_size = (fig_width_cm / 2.54, fig_height_cm / 2.54)  # Conversion to inches

                fig, axes = plt.subplots(2, 4, figsize=fig_size)  # Changed to 4x2 grid
                fig.suptitle('Mean absolute differences', fontsize=12)
                axes = axes.flatten()

                wrapper = textwrap.TextWrapper(width=15, break_long_words=False, break_on_hyphens=False)
                for idx, title in enumerate(model_titles):
                    ax = axes[idx]
                    im = ax.imshow(mean_diffs_dict[title], cmap='viridis', origin='upper', vmin=0, vmax=60)
                    wrapped_title = wrapper.fill(text=title)
                    ax.set_title(wrapped_title, fontsize=10)
                    ax.set_xticks(range(5))
                    ax.set_xticklabels(range(1, 6))
                    ax.set_yticks(range(13))
                    ax.set_yticklabels(range(13, 0, -1))

                for i in range(len(model_titles), len(axes)):
                    fig.delaxes(axes[i])

                # Use tight_layout first to avoid overlaps
                plt.tight_layout(rect=[0, 0, 0.9, 0.95])

                # Add a color bar
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                fig.colorbar(im, cax=cbar_ax)

                directory = f'plot_mean_differences_heatmap/{path}/'
                self._create_directory(directory)
                filename = f'{directory}All_models_heatmaps.pdf'
                filename = filename.replace(' ', '_')
                plt.savefig(filename, bbox_inches='tight',format='pdf')
                print(f'Saving to: {directory}All_models_heatmaps.pdf')
                plt.close()

    def plot_pearson_coefficient_grid(self):
        """
        Plot the Pearson coefficient of the correlation of one channel in the groundtruth signal with all other channels.
        """
        print("Start plotting Pearson coefficient grids")

        for data_dict in self.emg_data_dicts:
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original']
                model_titles = [title for title in datasets.keys() if title != 'Original']
                pearson_dict = {}

                for title, grid_2 in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    correlation_coefficients = np.zeros((13, 5))

                    for row in range(13):
                        for col in range(5):
                            correlation_coefficients[row, col], _ = pearsonr(groundtruth[row, col], grid_2[row, col])

                        pearson_dict[title] = correlation_coefficients

                        plt.figure()
                        plt.imshow(correlation_coefficients, cmap='viridis', origin='upper', vmin=0.95, vmax=1)
                        plt.xticks(range(5), labels=range(1, 6))
                        plt.yticks(range(13), labels=range(13, 0, -1))
                        plt.title(f'{title}')
                        plt.colorbar()

                        directory = f'plot_pearsonr_heatmap/{path}/'
                        self._create_directory(directory)
                        filename = f'{directory}{title}.pdf'
                        filename= filename.replace(' ', '_')
                        plt.savefig(filename, bbox_inches='tight',format='pdf')
                        print(f'Saving to: {directory}{title}.pdf')
                        plt.close()

                fig_width_cm = 16  # New width
                fig_height_cm = fig_width_cm * 0.9  # New height
                fig_size = (fig_width_cm / 2.54, fig_height_cm / 2.54)  # Conversion to inches

                fig, axes = plt.subplots(2, 4, figsize=fig_size)  # Changed to 4x2 grid
                fig.suptitle('Pearson coefficient', fontsize=12)
                axes = axes.flatten()

                wrapper = textwrap.TextWrapper(width=15, break_long_words=False, break_on_hyphens=False)
                for idx, title in enumerate(model_titles):
                    ax = axes[idx]
                    im = ax.imshow(pearson_dict[title], cmap='viridis', origin='upper', vmin=0.95, vmax=1)
                    wrapped_title = wrapper.fill(text=title)
                    ax.set_title(wrapped_title, fontsize=10)
                    ax.set_xticks(range(5))
                    ax.set_xticklabels(range(1, 6))
                    ax.set_yticks(range(13))
                    ax.set_yticklabels(range(13, 0, -1))

                for i in range(len(model_titles), len(axes)):
                    fig.delaxes(axes[i])

                # Use tight_layout first to avoid overlaps
                plt.tight_layout(rect=[0, 0, 0.9, 0.95])

                # Add a color bar
                cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                fig.colorbar(im, cax=cbar_ax)

                directory = f'plot_pearsonr_heatmap/{path}/'
                self._create_directory(directory)
                filename = f'{directory}All_models_pearson_heatmaps.pdf'
                filename = filename.replace(' ', '_')
                plt.savefig(filename, bbox_inches='tight',format='pdf')
                print(f'Saving to: {directory}All_models_pearson_heatmaps.pdf')
                plt.close()

    def plot_normalized_correlations(self):
        """
        Plot the normalized cross-correlation of the original signal with other signals.
        """
        print("Start plotting correlations")

        # Storage for mean and std calculations
        all_normalized_correlations = {}
        results = []

        # Custom colors
        mean_color = (47 / 255, 88 / 255, 110 / 255)  # RGB (47, 88, 110)
        std_color = (84 / 255, 156 / 255, 188 / 255)  # RGB (198, 222, 233)

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original']

                auto_corr = np.correlate(groundtruth[0, 0], groundtruth[0, 0], mode='full')
                max_corr = np.max(np.abs(auto_corr))
                max_corr_idx = np.argmax(np.abs(auto_corr))
                print(f"Max autocorrelation: {max_corr} at index {max_corr_idx} for {path}")

                # Initialize storage for correlations
                for title in datasets.keys():
                    if title != 'Original':
                        if title not in all_normalized_correlations:
                            all_normalized_correlations[title] = {}
                        if path not in all_normalized_correlations[title]:
                            all_normalized_correlations[title][path] = []

                # Iterate over datasets, skipping the 'Original' itself
                for title, emg_data in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    for row in range(13):
                        for col in range(5):
                            x = groundtruth[row, col, :]
                            y = emg_data[row, col, :]
                            mu_x = np.mean(x)
                            mu_y = np.mean(y)
                            std_x = np.std(x)
                            std_y = np.std(y)

                            if std_x == 0 or std_y == 0:
                                print(
                                    f"Standard deviation is zero for row {row}, col {col} in {title}. Skipping normalization.")
                                continue

                            # Normalize the signals
                            norm_x = (x - mu_x) / std_x
                            norm_y = (y - mu_y) / std_y

                            # Compute normalized cross-correlation
                            correlation = np.correlate(norm_x, norm_y, mode='full')
                            normalized_correlation = correlation / len(
                                x)  # Correct normalization by length of the signal overlap

                            if np.isnan(normalized_correlation).any():
                                print(f"NaN detected in normalized correlation for row {row}, col {col} in {title}.")
                                continue

                            max_corr = np.max(np.abs(normalized_correlation))
                            max_corr_idx = np.argmax(np.abs(normalized_correlation))
                            print(f"Max correlation: {max_corr} at index {max_corr_idx} for {path}")

                            # Store the normalized correlation
                            all_normalized_correlations[title][path].append(normalized_correlation)

            # Plot mean and std deviation plots for each title and each path
            for title, paths in all_normalized_correlations.items():
                for path, correlations in paths.items():
                    if len(correlations) > 0:
                        correlations_array = np.array(correlations)
                        mean_corr = np.nanmean(correlations_array, axis=0)
                        std_corr = np.nanstd(correlations_array, axis=0)
                        shifts = np.arange(-len(mean_corr) // 2 + 1, len(mean_corr) // 2 + 1)

                        # Replace NaNs with 0
                        mean_corr = np.nan_to_num(mean_corr)
                        std_corr = np.nan_to_num(std_corr)

                        fig, ax = plt.subplots(figsize=(16 / 2.54, 10 / 2.54))  # Convert cm to inches
                        ax.plot(shifts, mean_corr, label='Mean Correlation', color=mean_color)
                        ax.fill_between(shifts, mean_corr - std_corr, mean_corr + std_corr, color=std_color, alpha=0.2,
                                        label='Standard Deviation')

                        # Save the peak of the mean value plot
                        peak_index = int(np.argmax(mean_corr))
                        peak_mean = float(mean_corr[peak_index])
                        peak_std = float(std_corr[peak_index])
                        results.append({
                            'path': path,
                            'title': title,
                            'peak_index': peak_index,
                            'peak_mean': peak_mean,
                            'peak_std': peak_std
                        })

                        ax.set_xlabel('Shift')
                        ax.set_ylabel('Normalized Correlation')
                        ax.set_title('Mean Normalized Cross-Correlation with Std Dev')
                        ax.legend(loc='best')
                        ax.grid(True)
                        ax.set_ylim(-0.4, 1.1)
                        ax.set_xlim(-60000, 60000)
                        plt.tight_layout()

                        directory = f'plot_normalized_correlations/{title}/{path}/'
                        self._create_directory(directory)
                        filename = f'{directory}{title}_Mean_Normalized_Correlations.pdf'
                        filename = filename.replace(' ', '_')
                        plt.savefig(filename, format='pdf')
                        print(f'Saving summary plot for {title} - {path} to: {filename}')
                        plt.close()

            # Combine all mean and std plots into one figure for each path
            for path in data_dict.keys():
                combined_fig, combined_axes = plt.subplots(2, 4, figsize=(
                32 / 2.54, 16 / 2.54))  # 32x16 cm converted to inches
                combined_fig.suptitle('Mean and Std Dev Normalized Cross-Correlation', fontsize=12)
                combined_axes = combined_axes.flatten()

                for idx, (title, paths) in enumerate(all_normalized_correlations.items()):
                    if idx >= 8:
                        break
                    ax = combined_axes[idx]
                    if path in paths:
                        correlations = paths[path]
                        if len(correlations) > 0:
                            correlations_array = np.array(correlations)
                            mean_corr = np.nanmean(correlations_array, axis=0)
                            std_corr = np.nanstd(correlations_array, axis=0)
                            shifts = np.arange(-len(mean_corr) // 2 + 1, len(mean_corr) // 2 + 1)

                            mean_corr = np.nan_to_num(mean_corr)
                            std_corr = np.nan_to_num(std_corr)

                            ax.plot(shifts, mean_corr, label='Mean Correlation', color=mean_color)
                            ax.fill_between(shifts, mean_corr - std_corr, mean_corr + std_corr, color=std_color,
                                            alpha=0.2, label='Standard Deviation')
                            ax.set_title(f'{title}', fontsize=10)
                            ax.set_xlabel('Shift')
                            ax.set_ylabel('Normalized Correlation')
                            ax.grid(True)
                            ax.set_ylim(-0.4, 1.1)
                            ax.set_xlim(-60000, 60000)

                # Hide any unused subplots
                for j in range(idx + 1, len(combined_axes)):
                    combined_fig.delaxes(combined_axes[j])

                combined_fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)
                plt.tight_layout(rect=[0, 0, 1, 0.95])

                directory = 'plot_normalized_correlations/summary/'
                self._create_directory(directory)
                filename = f'{directory}Combined_Mean_Normalized_Correlations_{path}.pdf'
                filename = filename.replace(' ', '_')
                combined_fig.savefig(filename, format='pdf')
                print(f'Saving combined summary plot to: {filename}')
                plt.close()

        # Save the results to a JSON file in the plot_normalized_correlations folder
        results_path = 'plot_normalized_correlations/summary/metrics.json'
        self._create_directory('plot_normalized_correlations/summary/')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        print(f'Saving results to: {results_path}')

        print("Finished plotting all correlations and saving results.")

    def create_difference_plots(self):
        """
        Plot the absolute differences between the groundtruth signal and other signals.
        """
        print("Start plotting absolute difference grids")

        # Define the custom color
        plot_color = (47 / 255, 88 / 255, 110 / 255)  # RGB (47, 88, 110)
        positions = [(3, 1), (4, 2)]  # Only these two positions

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
                    fig.suptitle(f'Differences: Original vs {title}', fontsize=12)

                    # Additional figure for the large plot at position [1][4]
                    fig_large, ax_large = plt.subplots(figsize=(50, 8.27))  # A4 page in inches

                    for i in range(13):
                        for j in range(5):
                            ax = axes[i, j]
                            ax.plot(diff[i, j], color=plot_color)
                            ax.set_xticks([])
                            if j == 0:
                                y_ticks = np.arange(global_min - 50, global_max + 50, 500)
                                ax.set_yticks(y_ticks)
                                ax.set_yticklabels([f'{tick:.0f}' for tick in y_ticks])
                            else:
                                ax.set_yticks([])
                            ax.set_ylim(global_min, global_max)
                            if i == 1 and j == 4:
                                ax_large.plot(diff[i, j], color=plot_color)
                                ax_large.set_title(f'Large Plot: Differences at Row 2, Column 5 - {path}', fontsize=12)
                                ax_large.set_xlabel('Time Points')
                                ax_large.set_ylabel('Difference Value')
                                ax_large.set_ylim(global_min, global_max)
                                y_ticks_large = np.arange(global_min - 50, global_max + 50, 500)
                                ax_large.set_yticks(y_ticks_large)
                                ax_large.set_yticklabels([f'{tick:.0f}' for tick in y_ticks_large])

                    fig.tight_layout()
                    fig.subplots_adjust(top=0.95)
                    directory = f'plot_difference/{path}/'
                    self._create_directory(directory)
                    filename = f'{directory}Differences_{title}.pdf'
                    filename = filename.replace(' ', '_')
                    fig.savefig(filename, format='pdf')
                    print(f'Saving to: {filename}')
                    plt.close(fig)
                    fig_large.tight_layout()
                    fig_large.savefig(f'{directory}Differences_Large_Plot_Row_2_Column_5_{title}.pdf', format='pdf')
                    print(f'Saving large plot to: {directory}Differences_Large_Plot_Row_2_Column_5_{title}.pdf')
                    plt.close(fig_large)

                # Create figures for each position containing subplots for all models
                for (row, col) in positions:
                    fig_position, axes_position = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
                    axes_position = axes_position.flatten()

                    for idx, (title, grid_2) in enumerate(
                            {k: v for k, v in datasets.items() if k != 'Original'}.items()):
                        if idx >= 8:
                            break  # Ensure we don't create more than 8 subplots

                        diff = groundtruth - grid_2
                        ax = axes_position[idx]
                        ax.plot(diff[row, col], color=plot_color)
                        ax.set_title(f'{title}', fontsize=10)
                        ax.set_ylim(global_min, global_max)
                        ax.set_xlabel('Time Points')
                        ax.set_ylabel('Difference in µV')
                        ax.grid(True)

                    # Hide any unused subplots
                    for j in range(idx + 1, 8):
                        fig_position.delaxes(axes_position[j])

                    fig_position.tight_layout()
                    fig_position.subplots_adjust(top=0.9)

                    # Save the figure for each position
                    directory = f'plot_difference/{path}/positions/'
                    self._create_directory(directory)
                    filename = f'{directory}Differences_Position_{row + 1}_{col + 1}.pdf'
                    filename = filename.replace(' ', '_')
                    plt.savefig(filename, format='pdf')
                    print(f'Saving to: {filename}')
                    plt.close(fig_position)

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
                    fig.suptitle(f'Propagation Plot: {title} - {path}', fontsize=12)

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
                    filename = f'{directory}Propagation_Plot_{title}.pdf'
                    filename = filename.replace(' ', '_')
                    plt.savefig(filename,format='pdf')
                    print(f'Saving to: {directory}Propagation_Plot_{title}.pdf')
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

        # Define figure size
        fig_width_cm = 16
        fig_height_cm = 29.7  # A4 page height in cm
        fig_size = (fig_width_cm / 2.54, fig_height_cm / 2.54)  # Convert cm to inches

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

                    # Determine global min and max for y and x axes across all subplots
                    global_y_min, global_y_max = float('inf'), float('-inf')
                    global_x_min, global_x_max = float('inf'), float('-inf')

                    for row in range(13):
                        for col in range(5):
                            groundtruth_sig = groundtruth[row, col, :]
                            signal2 = emg_data[row, col, :]
                            correlations = self.calculate_windowed_correlation(groundtruth_sig, signal2, window_size)
                            x_axis = np.arange(len(correlations))

                            # Update global min and max for y and x axes
                            global_y_min = min(global_y_min, min(correlations))
                            global_y_max = max(global_y_max, max(correlations))
                            global_x_min = min(global_x_min, min(x_axis))
                            global_x_max = max(global_x_max, max(x_axis))

                    # Setup the figure for the big plot (13x5 grid of subplots)
                    fig, axes = plt.subplots(nrows=13, ncols=5, figsize=fig_size)

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
                            plt.title(f'Windowed Pearson Correlation - Channel ({row + 1}, {col + 1}) - {title}')
                            plt.ylim(global_y_min, global_y_max)
                            plt.xlim(global_x_min, global_x_max)
                            filename = f'{directory}Windowed_Correlation_Channel_{row + 1}_{col + 1}.pdf'
                            filename = filename.replace(' ', '_')
                            plt.savefig(f'{directory}Windowed_Correlation_Channel_{row + 1}_{col + 1}.pdf')
                            plt.close()

                            # Add to big subplot figure
                            ax = axes[row, col]
                            ax.plot(x_axis, correlations)
                            ax.set_title(f'Channel ({row + 1}, {col + 1})')
                            ax.set_ylim(global_y_min, global_y_max)
                            ax.set_xlim(global_x_min, global_x_max)
                            ax.label_outer()  # Only show outer labels to reduce clutter

                    # Save the large figure with all subplots
                    fig.tight_layout()
                    fig.subplots_adjust(top=0.95)
                    fig.suptitle(f'Windowed Pearson Correlation - {title}')
                    filename = f'{directory}Combined_Windowed_Correlation_{title}.pdf'
                    filename = filename.replace(' ', '_')
                    fig.savefig(filename)
                    print(f'Saving to: {directory}Combined_Windowed_Correlation_{title}.pdf')

                    plt.close(fig)

                    # Plot differences if the current title is not 'Bilinear interpolation'
                    if title != 'Bilinear interpolation':
                        diff_directory = f'plot_windowed_correlations/{path}/Linear_vs_{title}/'
                        self._create_directory(diff_directory)

                        # Determine global min and max for differences across all subplots
                        global_diff_y_min, global_diff_y_max = float('inf'), float('-inf')
                        global_diff_x_min, global_diff_x_max = float('inf'), float('-inf')

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

                                # Update global min and max for differences
                                global_diff_y_min = min(global_diff_y_min, min(differences))
                                global_diff_y_max = max(global_diff_y_max, max(differences))
                                global_diff_x_min = min(global_diff_x_min, min(x_axis_diff))
                                global_diff_x_max = max(global_diff_x_max, max(x_axis_diff))

                        fig_diff, axes_diff = plt.subplots(nrows=13, ncols=5, figsize=fig_size)

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
                                    f'Difference in Windowed Pearson Correlation - Bilinear interpolation vs {title}')
                                plt.ylim(global_diff_y_min, global_diff_y_max)
                                plt.xlim(global_diff_x_min, global_diff_x_max)
                                filename = f'{diff_directory}Difference_Windowed_Correlation_Channel_{row + 1}_{col + 1}.pdf'
                                filename = filename.replace(' ', '_')
                                plt.savefig(
                                    filename)
                                plt.close()

                                # Add to big subplot figure for differences
                                ax_diff = axes_diff[row, col]
                                ax_diff.fill_between(x_axis_diff, differences, where=(differences >= 0),
                                                     color='lightgreen', alpha=0.5)
                                ax_diff.fill_between(x_axis_diff, differences, where=(differences < 0),
                                                     color='lightcoral', alpha=0.5)
                                ax_diff.plot(x_axis_diff, differences, color='black')
                                ax_diff.set_ylim(global_diff_y_min, global_diff_y_max)
                                ax_diff.set_xlim(global_diff_x_min, global_diff_x_max)
                                ax_diff.label_outer()

                        # Save the large difference figure
                        fig_diff.tight_layout()
                        fig_diff.subplots_adjust(top=0.95)
                        fig_diff.suptitle(f'Difference in Windowed Pearson Correlation - Bilinear interpolation vs {title}')
                        filename = f'{diff_directory}Combined_Difference_{title}.pdf'
                        filename = filename.replace(' ', '_')
                        fig_diff.savefig(filename)
                        print(f'Saving to: {diff_directory}Combined_Difference_{title}.pdf')

                        plt.close(fig_diff)

        print("Finished plotting all correlations and differences.")


    def create_box_plot(self):
        data_dicts = self._load_data()
        self._create_directory('boxplots')

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
                        even_diffs = np.abs(model_data[::2, ::2] - origin_hr[::2, ::2])
                        differences_even[model].extend(even_diffs.flatten())

                        # Odd columns completely and odd rows in the even columns
                        odd_cols_completely = model_data[:, 1::2]
                        odd_rows_even_cols = model_data[1::2, ::2]
                        odd_cols_completely_orig = origin_hr[:, 1::2]
                        odd_rows_even_cols_orig = origin_hr[1::2, ::2]

                        odd_diffs_cols_completely = np.abs(odd_cols_completely - odd_cols_completely_orig).flatten()
                        odd_diffs_rows_even_cols = np.abs(odd_rows_even_cols - odd_rows_even_cols_orig).flatten()

                        differences_odd[model].extend(odd_diffs_cols_completely)
                        differences_odd[model].extend(odd_diffs_rows_even_cols)

        def calculate_outliers(data):
            if len(data) == 0:
                return 0, data  # Return 0 outliers and the original empty data

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

        def save_metrics(differences, filename):
            metrics = {}
            for model, diffs in differences.items():
                metrics[model] = {
                    'Mean': np.mean(diffs),
                    'Std': np.std(diffs),
                    'Q1': np.percentile(diffs, 25),
                    'Median': np.median(diffs),
                    'Q3': np.percentile(diffs, 75),
                    'Min': np.min(diffs),
                    'Max': np.max(diffs)
                }
            df_metrics = pd.DataFrame(metrics).transpose()
            df_metrics.to_csv(filename)

        def plot_grouped_boxplot(differences_all, differences_even, differences_odd, title, colors, filename,
                                 remove_outliers=False, group_spacing=2, box_spacing=0, box_width=0.7):
            boxplot_data = []
            labels = []  # Will hold labels for the x-axis, which are model names wrapped if needed

            # Creating a TextWrapper object to wrap text without breaking words
            wrapper = textwrap.TextWrapper(width=15, break_long_words=False, break_on_hyphens=False)

            for model in differences_all.keys():
                all_diffs = differences_all[model]
                even_diffs = differences_even[model]
                odd_diffs = differences_odd[model]
                if remove_outliers:
                    _, all_diffs = calculate_outliers(all_diffs)
                    _, even_diffs = calculate_outliers(even_diffs)
                    _, odd_diffs = calculate_outliers(odd_diffs)

                boxplot_data.append(all_diffs)
                boxplot_data.append(even_diffs)
                boxplot_data.append(odd_diffs)
                labels.append(wrapper.fill(text=model))

            fig, ax = plt.subplots(figsize=(7, 4))
            positions = []
            for i in range(len(differences_all)):
                base_position = i * (3 * box_width + group_spacing)
                positions.extend([base_position + j * (box_width + box_spacing) for j in range(3)])

            bp = ax.boxplot(boxplot_data, positions=positions, patch_artist=True, widths=box_width)

            # Color the boxes
            for i, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[i % len(colors)])

            # Add mean cross markers
            for i, data in enumerate(boxplot_data):
                mean = np.mean(data)
                ax.plot(positions[i], mean, marker='x', color='black', markersize=4, markeredgewidth=1)

            # Create a legend
            legend_elements = [
                plt.Line2D([0], [0], color=colors[0], lw=4, label='All\nchannels'),
                plt.Line2D([0], [0], color=colors[1], lw=4, label='Predicted\nchannel'),
                plt.Line2D([0], [0], color=colors[2], lw=4, label='Given\nchannel'),
                plt.Line2D([0], [0], marker='x', color='w', label='Mean', markerfacecolor='red', markersize=8,
                           markeredgewidth=2)
            ]

            # Adjust the ylim to give more space for outlier percentages
            y_max = max([max(data) for data in boxplot_data if data]) * 1.06  # Increase by 6% for space
            ax.set_ylim(0, y_max)

            # Setting title with bold font and adjusting spacing
            ax.set_title(title, pad=20, fontsize=12)

            plt.xlabel('Models')
            plt.ylabel('Absolute difference in µV')
            ax.set_xticks([np.mean(positions[i * 3:(i + 1) * 3]) for i in range(len(differences_all))])
            ax.set_xticklabels(labels, rotation=0, ha='center')
            plt.grid(True)
            plt.tight_layout()

            # Place the legend outside the plot in the top right corner
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fancybox=False, shadow=False)
            filename = filename.replace(' ', '_')
            plt.savefig(filename, bbox_inches='tight', format='png', dpi=300)
            print("Automatically chosen figure size: Width = {:.2f} inches, Height = {:.2f} inches".format(
                fig.get_figwidth(), fig.get_figheight()))
            plt.close()

        colors = ['#C6DEE9', '#5DC08A', '#2F586E']  # Different colors for each data type group.

        # Plot grouped boxplot without outliers
        plot_grouped_boxplot(differences_all, differences_even, differences_odd,
                             'Absolute difference between real and synthetic data (Without outliers)',
                             colors,
                             'boxplots/boxplot_grouped_data_no_outliers.png', remove_outliers=True)

        # Plot grouped boxplot with outliers
        plot_grouped_boxplot(differences_all, differences_even, differences_odd,
                             'Absolute difference between real and synthetic data (With outliers)', colors,
                             'boxplots/boxplot_grouped_data_with_outliers.png', remove_outliers=False)

        save_metrics(differences_all, 'boxplots/metrics_all_data.csv')
        save_metrics(differences_even, 'boxplots/metrics_given_channels.csv')
        save_metrics(differences_odd, 'boxplots/metrics_predicted_channels.csv')

    def plot_specific_positions(self):
        """
        Create plots for specific positions [3][1] and [4][2] for each path and selected models.
        """
        positions = [(3, 1), (4, 2)]
        x_range = slice(30000, 30128)  # x-axis range from 30000 to 30128 (128 samples)

        models = [
            'Original', 'Bilinear interpolation', 'Bicubic spline interpolation', 'SRGAN',
            'UNet', 'UNet with Fourier', 'UNet with Wavelet', 'CNN-LSTM'
        ]

        # Define colors matching the order of the models
        colors = [
            (148 / 255, 11 / 255, 45 / 255), (127 / 255, 127 / 255, 127 / 255),
            (47 / 255, 88 / 255, 110 / 255), (255 / 255, 192 / 255, 0 / 255),
            (239 / 255, 167 / 255, 141 / 255), (0 / 255, 176 / 255, 240 / 255),
            (112 / 255, 48 / 255, 160 / 255), (0 / 255, 128 / 255, 0 / 255)
        ]

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                # Create a figure with two subplots
                fig, axs = plt.subplots(1, 2,
                                        figsize=(20 / 2.54, 14 / 2.54))  # Increase width to 24 cm and height to 14 cm

                # Iterate over specified positions
                for i, (row, col) in enumerate(positions):
                    ax = axs[i]
                    for color, title in zip(colors, models):
                        if title in datasets:
                            alpha = 1 if title == 'Original' else 0.8
                            signal = datasets[title][row, col, x_range]
                            ax.plot(signal, label=title, alpha=alpha, color=color)
                    ax.set_xlabel('Time in samples')
                    ax.set_ylabel('Amplitude in µV')
                    subtitle = 'predicted channel' if row == 3 else 'given channel'
                    ax.set_title(f'sEMG - {subtitle}')

                # Add a single legend in the middle below the two subplots
                handles, labels = axs[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2)

                plt.tight_layout()

                # Create directory and save the plot
                directory = f'plot_specific_positions/{path}/'
                self._create_directory(directory)
                filename = f'{directory}Position_3_1_and_4_2.pdf'
                filename = filename.replace(' ', '_')
                plt.savefig(filename, bbox_inches='tight', format='pdf')
                plt.close()

                print(f'Saving to: {directory}Position_3_1_and_4_2.pdf')

    def plot_frequency_spectrum_difference(self):
        """
        Plot the differences in the frequency spectrum between the groundtruth signal and other signals.
        """
        print("Start plotting frequency spectrum differences")

        # Function to compute the Fourier Transform and frequency spectrum
        def compute_frequency_spectrum(signal, fs=2048):
            n = len(signal)
            freq = np.fft.fftfreq(n, d=1 / fs)
            spectrum = np.fft.fft(signal)
            return freq, np.abs(spectrum)

        positions = [(3, 1), (4, 2)]  # Only these two positions

        # Custom colors
        spectrum_color = (47 / 255, 88 / 255, 110 / 255)  # RGB (47, 88, 110)
        difference_color = (84 / 255, 156 / 255, 188 / 255)

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original']

                for (row, col) in positions:
                    # Create a figure for each position with 8 subplots (2 rows, 4 columns)
                    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
                    fig.tight_layout(pad=5.0)

                    # Flatten axes array for easier iteration
                    axes = axes.flatten()

                    # Iterate over datasets, skipping the 'Original' itself
                    for i, (title, emg_data) in enumerate(
                            {k: v for k, v in datasets.items() if k != 'Original'}.items()):
                        if i >= 8:
                            break  # Ensure we don't create more than 8 subplots

                        groundtruth_sig = groundtruth[row, col, :]
                        signal2 = emg_data[row, col, :]

                        freq_gt, spectrum_gt = compute_frequency_spectrum(groundtruth_sig)
                        freq_sig, spectrum_sig = compute_frequency_spectrum(signal2)

                        spectrum_diff = spectrum_gt - spectrum_sig

                        # Plot original spectrum
                        axes[i].plot(freq_gt, spectrum_gt, label='Original Spectrum', color=spectrum_color)
                        # Plot spectrum difference
                        axes[i].plot(freq_gt, spectrum_diff, label='Spectrum Difference', color=difference_color)
                        axes[i].set_xlim(0, 600)  # Limiting the frequency range for better visualization
                        axes[i].set_title(f'{title}')
                        axes[i].set_xlabel('Frequency (Hz)')
                        axes[i].set_ylabel('Magnitude')
                        axes[i].grid(True)
                        axes[i].legend()

                        # Save the individual subplot
                        directory = f'plot_frequency_spectrum_difference/{path}/individual_subplots/'
                        self._create_directory(directory)
                        subplot_save_path = f'{directory}{title}_Frequency_Spectrum_Comparison_Row_{row + 1}_Col_{col + 1}.pdf'
                        subplot_save_path = subplot_save_path.replace(' ', '_')
                        fig_subplot, ax_subplot = plt.subplots()
                        ax_subplot.plot(freq_gt, spectrum_gt, label='Original Spectrum', color=spectrum_color)
                        ax_subplot.plot(freq_gt, spectrum_diff, label='Spectrum Difference', color=difference_color)
                        ax_subplot.set_xlim(0, 600)
                        ax_subplot.set_title(f'{title}')
                        ax_subplot.set_xlabel('Frequency (Hz)')
                        ax_subplot.set_ylabel('Magnitude')
                        ax_subplot.grid(True)
                        ax_subplot.legend()
                        plt.savefig(subplot_save_path, format='pdf')
                        plt.close(fig_subplot)

                    # Remove any unused subplots
                    for j in range(i + 1, 8):
                        fig.delaxes(axes[j])

                    # Save the combined figure for the current position
                    directory = f'plot_frequency_spectrum_difference/{path}/'
                    self._create_directory(directory)
                    save_path = f'{directory}Frequency_Spectrum_Comparison_Row_{row + 1}_Col_{col + 1}.pdf'
                    save_path = save_path.replace(' ', '_')
                    plt.savefig(save_path, format='pdf')
                    print(f'Saving to: {save_path}')
                    plt.close(fig)

        print("Finished plotting frequency spectrum differences.")


if __name__ == '__main__':
    file_path = '../results/'
    visualize = Visualizer(file_path)
    # visualize.plot_emg_channels()
    #visualize.plot_emg_channels_difference()
    # visualize.plot_pearson_coefficient_grid()
    #visualize.create_difference_plots()
    # visualize.propagation_plot()
    visualize.plot_windowed_correlations()
    #
    # visualize.create_box_plot()
    #visualize.plot_normalized_correlations()
    # visualize.plot_specific_positions()
    #visualize.plot_frequency_spectrum_difference()

    #TODO: Time in s umschreiben