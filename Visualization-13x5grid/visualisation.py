"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Visualizer, which is used to visualize the results of the interpolation
and the neural networks.

The code is started by calling the main function that creates an object of the Visualizer class. The object is then used
to call the different visualization functions.
"""
import csv
import os
import textwrap

import numpy as np
import matplotlib
from matplotlib.ticker import MultipleLocator

matplotlib.use('Agg')
import pandas as pd

import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
import json

from scipy.io import loadmat



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
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

        # Update rcParams for global settings
        plt.rcParams['axes.titlesize'] = 10  # Subplot titles
        plt.rcParams['axes.labelsize'] = 10  # Axis labels
        plt.rcParams['figure.titlesize'] = 12  # Suptitle
        plt.rcParams['legend.fontsize'] = 10  # Legend
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

        self.file_path = file_path
        self.emg_data_dicts = self._load_data()
        self.page_width = 16/2.54
        self.page_height = 22/2.54
        print("Initialization done")

    def _load_data(self):
        paths = ['path0', 'path1', 'path2', 'path3']
        dim_sizes = {
            'path0': 64576,
            'path1': 60736,
            'path2': 60736,
            'path3': 65984
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
            interpolation_lr_sr_filtered = apply_bandpass_filter(interpolation_lr_sr)
            cubic_lr_sr_filtered = apply_bandpass_filter(cubic_lr_sr)
            model1_lr_sr_filtered = apply_bandpass_filter(model1_lr_sr)
            model2_lr_sr_filtered = apply_bandpass_filter(model2_lr_sr)
            model2_Fourier_lr_sr_filtered = apply_bandpass_filter(model2_Fourier_lr_sr)
            model2_Wavelet_lr_sr_filtered = apply_bandpass_filter(model2_Wavelet_lr_sr)
            model3_lr_sr_filtered = apply_bandpass_filter(model3_lr_sr)
            origin_hr_filtered = apply_bandpass_filter(origin_hr)

            # Plot before and after filtering
            self.plot_channel_3_2(path, 'Bilinear', interpolation_lr_sr, interpolation_lr_sr_filtered)
            self.plot_channel_3_2(path, 'Bicubic spline', cubic_lr_sr, cubic_lr_sr_filtered)
            self.plot_channel_3_2(path, 'SRGAN', model1_lr_sr, model1_lr_sr_filtered)
            self.plot_channel_3_2(path, 'UNet', model2_lr_sr, model2_lr_sr_filtered)
            self.plot_channel_3_2(path, 'UNet Fourier', model2_Fourier_lr_sr, model2_Fourier_lr_sr_filtered)
            self.plot_channel_3_2(path, 'UNet Wavelet', model2_Wavelet_lr_sr, model2_Wavelet_lr_sr_filtered)
            self.plot_channel_3_2(path, 'CNN-LSTM', model3_lr_sr, model3_lr_sr_filtered)
            self.plot_channel_3_2(path, 'Original', origin_hr, origin_hr_filtered)

            # Create a dictionary with the loaded and adjusted data
            emg_data_dict = {
                path: {  # Associate each dataset with its path
                    'Original': origin_hr_filtered,
                    'Bilinear': interpolation_lr_sr_filtered,
                    'Bicubic spline': cubic_lr_sr_filtered,
                    'SRGAN': model1_lr_sr_filtered,
                    'UNet': model2_lr_sr_filtered,
                    'UNet Fourier': model2_Fourier_lr_sr_filtered,
                    'UNet Wavelet': model2_Wavelet_lr_sr_filtered,
                    'CNN-LSTM': model3_lr_sr_filtered
                }
            }

            data_dicts.append(emg_data_dict)

        return data_dicts

    def plot_channel_3_2(self, path, model_name, original_data, filtered_data):
        start_idx = 30050
        end_idx = start_idx + 256
        channel_original = original_data[3, 2, start_idx:end_idx]
        channel_filtered = filtered_data[3, 2, start_idx:end_idx]

        plt.figure(figsize=(10, 4))
        plt.plot(channel_original, label='Original', color='blue')
        plt.plot(channel_filtered, label='Filtered', color='red')
        plt.title(f'{model_name} - Channel [3][2]')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()

        directory = f'plot_channel_3_2/{path}/'
        self._create_directory(directory)
        filename = f'{directory}{model_name}_Channel_3_2_comparison.pdf'
        plt.savefig(filename, bbox_inches='tight', format='pdf')
        plt.close()
        print(f'Saving plot to: {filename}')

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

    # def _load_matlab_data(self):
    #     test_paths = [
    #         '../data/Alessandros Daten sEMG/Test/RPAR_DECOMP_PROX_35-50-70_FINISHED/POST/400PaRiRo170628134308 Rampa_35MVC_5MVCs_2acontrazione_TODECOMPOSE_offset0_length30_runs250_EDITED_MATTEO.mat',
    #         '../data/Alessandros Daten sEMG/Test/RPAR_DECOMP_PROX_35-50-70_FINISHED/PRE/400PaRifd170606133208 Rampa_35MVC_5MVCs_2acontrazione_TODECOMPOSE_offset0_length27_runs250_FINISHED_MATTEO.mat',
    #         '../data/Alessandros Daten sEMG/Test/SNUC_DECOMP_PROX_35-50-70_FINISHED/POST/400NuStRo171221103923 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length29.665_runs100_FINISHED.mat',
    #         '../data/Alessandros Daten sEMG/Test/SNUC_DECOMP_PROX_35-50-70_FINISHED/PRE/400NuStRo171215104010 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs100_FINISHED.mat'
    #     ]
    #
    #     path_to_sample_position = {}
    #     for path, test_path in zip(['path0', 'path1', 'path2', 'path3'], test_paths):
    #         mat_data = loadmat(test_path)
    #         sample_position = mat_data['MUPulses'][0, 0][0, 0]  # Adjust as necessary based on actual structure
    #         path_to_sample_position[path] = sample_position
    #
    #     return path_to_sample_position
    #
    # def _calculate_rms(self, data, center_position, window_size=50, fs=2048):
    #     half_window = int(window_size * fs / 1000 / 2)
    #     start = center_position - half_window
    #     end = center_position + half_window
    #     rms_values = np.sqrt(np.mean(np.square(data[:, :, start:end]), axis=2))
    #     return rms_values
    #
    # def process_matlab_rms(self):
    #     path_to_sample_position = self._load_matlab_data()
    #     window_size = 50  # in ms
    #     fs = 2048  # Sampling frequency
    #
    #     for data_dict in self.emg_data_dicts:
    #         for path, datasets in data_dict.items():
    #             sample_position = path_to_sample_position.get(path)
    #             if sample_position is not None:
    #                 rms_dict = {}
    #                 for title, emg_data in datasets.items():
    #                     rms_values = self._calculate_rms(emg_data, sample_position, window_size, fs)
    #                     rms_dict[title] = rms_values
    #
    #                 self.plot_rms_heatmap(rms_dict, path, sample_position)
    #
    #                 # Processing for MUPulses{1,2}
    #                 mu_pulses_rms_dict = {}
    #                 for title, emg_data in datasets.items():
    #                     mu_pulses_rms_values = self._calculate_rms(emg_data, sample_position, window_size, fs)
    #                     mu_pulses_rms_dict[title] = mu_pulses_rms_values
    #
    #                 self.plot_rms_heatmap(mu_pulses_rms_dict, path, sample_position, mu_pulses=True)

    # def _calculate_rms(self, data, spike_positions, window_size=50, fs=2048):
    #     half_window = int(window_size * fs / 1000 / 2)
    #     num_channels = data.shape[0]
    #     rms_values = np.zeros((num_channels, num_channels))
    #
    #     for row in range(num_channels):
    #         for col in range(num_channels):
    #             signal = data[row, col, :]
    #             chnl_average = np.zeros(2 * half_window + 1)
    #
    #             for spike in spike_positions:
    #                 start = int(spike - half_window)
    #                 end = int(spike + half_window)
    #                 if start >= 0 and end < len(signal):
    #                     chnl_average += signal[start:end + 1]
    #
    #             chnl_average /= len(spike_positions)
    #             rms_values[row, col] = np.sqrt(np.mean(np.square(chnl_average)))
    #
    #     return rms_values
    #
    # def process_matlab_rms(self):
    #     path_to_sample_position = self._load_matlab_data()
    #     window_size = 50  # in ms
    #     fs = 2048  # Sampling frequency
    #
    #     for data_dict in self.emg_data_dicts:
    #         for path, datasets in data_dict.items():
    #             sample_position = path_to_sample_position.get(path)
    #             if sample_position is not None:
    #                 rms_dict = {}
    #                 for title, emg_data in datasets.items():
    #                     rms_values = self._calculate_rms(emg_data, sample_position, window_size, fs)
    #                     rms_dict[title] = rms_values
    #
    #                 self.plot_rms_heatmap(rms_dict, path, sample_position)
    #
    #                 # Processing for MUPulses{1,2}
    #                 mu_pulses_rms_dict = {}
    #                 for title, emg_data in datasets.items():
    #                     mu_pulses_rms_values = self._calculate_rms(emg_data, sample_position, window_size, fs)
    #                     mu_pulses_rms_dict[title] = mu_pulses_rms_values
    #
    #                 self.plot_rms_heatmap(mu_pulses_rms_dict, path, sample_position, mu_pulses=True)
    #
    # def plot_rms_heatmap(self, rms_dict, path, sample_position, mu_pulses=False):
    #     plt.figure(figsize=(12, 8))
    #     for i, (title, rms_values) in enumerate(rms_dict.items()):
    #         ax = plt.subplot(2, 4, i + 1)
    #         im = ax.imshow(rms_values, cmap='viridis', aspect='auto')
    #         plt.colorbar(im, ax=ax, label='RMS')
    #         ax.set_title(f'{title}')
    #         ax.set_xlabel('Channel')
    #         ax.set_ylabel('Channel')
    #
    #         # Set custom ticks and labels
    #         ax.set_xticks(range(8))
    #         ax.set_xticklabels(range(1, 9))
    #         ax.set_yticks(range(8))
    #         ax.set_yticklabels(range(1, 9))
    #
    #     suptitle = f'RMS, 50 ms window'
    #     if mu_pulses:
    #         suptitle += ' (MUPulses{1,2})'
    #
    #     plt.suptitle(suptitle)
    #     plt.tight_layout(rect=[0, 0, 1, 0.96])
    #
    #     directory = f'plot_rms_heatmap/{path}/'
    #     self._create_directory(directory)
    #     filename_suffix = '_MUPulses' if mu_pulses else ''
    #     filename = f'{directory}RMS_Heatmap_{sample_position}{filename_suffix}.pdf'
    #     filename = filename.replace(' ', '_')
    #     plt.savefig(filename, format='pdf')
    #     plt.close()
    #     print(f'Saved RMS heatmap to: {filename}')

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
                            # Create a time array
                            time_array = np.arange(len(signal)) / 2048.0
                            plt.plot(time_array, signal.squeeze() + offset,
                                     label=f'Channel row {rows + 1} col {cols + 1}')
                            offset += 3500  # Adjust offset for visual clarity

                    plt.xlabel('Time in s')
                    plt.title(f'{title}')
                    handles, labels = plt.gca().get_legend_handles_labels()
                    plt.legend(handles[::-1], labels[::-1], loc='upper left', bbox_to_anchor=(1, 1))
                    plt.tight_layout()

                    directory = f'plot_channels/{path}/'
                    self._create_directory(directory)
                    file_name = f'{directory}{title}.pdf'
                    file_name = file_name.replace(' ', '_')
                    plt.savefig(file_name, format='pdf')
                    print(f'Saving to: {directory}{title}.pdf')
                    plt.close()

    def plot_emg_channels_difference(self):
        """
        Plot the mean absolute error between the channels of the groundtruth signal and other signals.
        """
        print("Start plotting EMG differences")

        overall_mean_diffs_dict = {}
        path_count = 0

        for data_dict in self.emg_data_dicts:
            path_count += 1
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original']
                model_titles = [title for title in datasets.keys() if title != 'Original']
                mean_diffs_dict = {}

                for title, emg_data in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    mean_absolute_differences = np.mean(np.abs(groundtruth - emg_data), axis=2)
                    mean_diffs_dict[title] = mean_absolute_differences

                    # Add to overall mean differences dictionary
                    if title not in overall_mean_diffs_dict:
                        overall_mean_diffs_dict[title] = mean_absolute_differences
                    else:
                        overall_mean_diffs_dict[title] += mean_absolute_differences

                    plt.figure()
                    offset = 0
                    for row in range(13):
                        for col in range(5):
                            diff = np.abs(groundtruth[row, col, :] - emg_data[row, col, :])
                            time_array = np.arange(len(diff)) / 2048.0
                            plt.plot(time_array, diff + offset, label=f'Channel row {row + 1} col {col + 1}')
                            offset += np.max(diff) + 1000

                    plt.xlabel('Time in s')
                    plt.ylabel('Mean absolute error (with offset)')
                    plt.title(f'{title}')
                    plt.legend(loc='upper right', ncol=2)
                    plt.tight_layout(rect=[0, 0, 1, 0.95])

                    directory = f'plot_channels_difference/{path}/'
                    self._create_directory(directory)
                    filename = f'{directory}{title}_channels_difference.pdf'
                    filename = filename.replace(' ', '_')
                    plt.savefig(filename, format='pdf')
                    print(f'Saving to: {directory}{title}_channels_difference.pdf')
                    plt.close()

                    plt.figure()
                    plt.imshow(mean_absolute_differences, cmap='viridis', origin='upper', vmin=0, vmax=60)
                    plt.xticks(range(5), labels=range(1, 6))
                    plt.yticks(range(13), labels=range(13, 0, -1))
                    plt.title(f'Mean absolute error - {title}')
                    plt.colorbar()

                    directory = f'plot_mean_differences_heatmap/{path}/'
                    self._create_directory(directory)
                    filename = f'{directory}{title}_mean_differences.pdf'
                    filename = filename.replace(' ', '_')
                    plt.savefig(filename, bbox_inches='tight', format='pdf')
                    print(f'Saving to: {directory}{title}_mean_differences.pdf')
                    plt.close()

                fig_size = (self.page_width, self.page_height * 0.6)  # Conversion to inches

                fig, axes = plt.subplots(2, 4, figsize=fig_size)  # Changed to 4x2 grid
                fig.suptitle('Mean absolute error in µV')
                axes = axes.flatten()

                wrapper = textwrap.TextWrapper(width=15, break_long_words=False, break_on_hyphens=False)
                for idx, title in enumerate(model_titles):
                    ax = axes[idx]
                    im = ax.imshow(mean_diffs_dict[title], cmap='viridis', origin='upper', vmin=0, vmax=60)
                    wrapped_title = wrapper.fill(text=title)
                    ax.set_title(wrapped_title)
                    ax.set_xticks(range(5))
                    ax.set_xticklabels(range(1, 6))
                    ax.set_yticks(range(13))
                    ax.set_yticklabels(range(13, 0, -1))
                    ax.set_xlabel('Channel')
                    ax.set_ylabel('Channel')

                for i in range(len(model_titles), len(axes)):
                    fig.delaxes(axes[i])

                # Use tight_layout first to avoid overlaps
                plt.tight_layout(rect=[0, 0, 0.9, 0.99])

                # Add a color bar
                cbar_ax = fig.add_axes([0.92, 0.06, 0.02, 0.8])
                fig.colorbar(im, cax=cbar_ax)

                directory = f'plot_mean_differences_heatmap/{path}/'
                self._create_directory(directory)
                filename = f'{directory}All_models_heatmaps.pdf'
                filename = filename.replace(' ', '_')
                plt.savefig(filename, bbox_inches='tight', format='pdf')
                print(f'Saving to: {directory}All_models_heatmaps.pdf')
                plt.close()

        # Calculate the mean across all paths
        for title in overall_mean_diffs_dict:
            overall_mean_diffs_dict[title] /= path_count

        fig_size = (self.page_width, self.page_height * 0.6)  # Conversion to inches

        fig, axes = plt.subplots(2, 4, figsize=fig_size)  # Changed to 4x2 grid
        fig.suptitle('Mean absolute error')
        axes = axes.flatten()

        wrapper = textwrap.TextWrapper(width=15, break_long_words=False, break_on_hyphens=False)
        for idx, title in enumerate(model_titles):
            ax = axes[idx]
            im = ax.imshow(overall_mean_diffs_dict[title], cmap='viridis', origin='upper', vmin=0, vmax=60)
            wrapped_title = wrapper.fill(text=title)
            ax.set_title(wrapped_title)
            ax.set_xticks(range(5))
            ax.set_xticklabels(range(1, 6))
            ax.set_yticks(range(13))
            ax.set_yticklabels(range(13, 0, -1))
            ax.set_xlabel('Channel')
            ax.set_ylabel('Channel')

        for i in range(len(model_titles), len(axes)):
            fig.delaxes(axes[i])

        # Use tight_layout first to avoid overlaps
        plt.tight_layout(rect=[0, 0, 0.9, 0.99])

        # Add a color bar
        cbar_ax = fig.add_axes([0.92, 0.06, 0.02, 0.8])
        fig.colorbar(im, cax=cbar_ax)

        directory = f'plot_mean_differences_heatmap/summary/'
        self._create_directory(directory)
        filename = f'{directory}All_models_heatmaps_all_paths.pdf'
        filename = filename.replace(' ', '_')
        plt.savefig(filename, bbox_inches='tight', format='pdf')
        print(f'Saving to: {directory}All_models_heatmaps_all_paths.pdf')
        plt.close()

    def plot_pearson_coefficient_grid(self):
        """
        Plot the Pearson coefficient of the correlation of one channel in the groundtruth signal with all other channels.
        """
        print("Start plotting Pearson coefficient grids")

        overall_pearson_dict = {}
        path_count = 0

        for data_dict in self.emg_data_dicts:
            path_count += 1
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

                    # Add to overall pearson dictionary
                    if title not in overall_pearson_dict:
                        overall_pearson_dict[title] = np.zeros((13, 5))
                    overall_pearson_dict[title] += correlation_coefficients

                    plt.figure()
                    plt.imshow(correlation_coefficients, cmap='viridis', origin='upper', vmin=0.95, vmax=1)
                    plt.xticks(range(5), labels=range(1, 6))
                    plt.yticks(range(13), labels=range(13, 0, -1))
                    plt.title(f'{title}')
                    plt.colorbar()

                    directory = f'plot_pearsonr_heatmap/{path}/'
                    self._create_directory(directory)
                    filename = f'{directory}{title}.pdf'
                    filename = filename.replace(' ', '_')
                    plt.savefig(filename, bbox_inches='tight', format='pdf')
                    print(f'Saving to: {directory}{title}.pdf')
                    plt.close()

                fig_size = (self.page_width, self.page_height * 0.6)  # Conversion to inches

                fig, axes = plt.subplots(2, 4, figsize=fig_size)  # Changed to 4x2 grid
                fig.suptitle('Pearson coefficient')
                axes = axes.flatten()

                wrapper = textwrap.TextWrapper(width=15, break_long_words=False, break_on_hyphens=False)
                for idx, title in enumerate(model_titles):
                    ax = axes[idx]
                    im = ax.imshow(pearson_dict[title], cmap='viridis', origin='upper', vmin=0.95, vmax=1)
                    wrapped_title = wrapper.fill(text=title)
                    ax.set_title(wrapped_title)
                    ax.set_xticks(range(5))
                    ax.set_xticklabels(range(1, 6))
                    ax.set_yticks(range(13))
                    ax.set_yticklabels(range(13, 0, -1))
                    ax.set_xlabel('Channel')
                    ax.set_ylabel('Channel')

                for i in range(len(model_titles), len(axes)):
                    fig.delaxes(axes[i])

                # Use tight_layout first to avoid overlaps
                plt.tight_layout(rect=[0, 0, 0.9, 0.99])

                # Add a color bar
                cbar_ax = fig.add_axes([0.92, 0.06, 0.02, 0.8])
                fig.colorbar(im, cax=cbar_ax)

                directory = f'plot_pearsonr_heatmap/{path}/'
                self._create_directory(directory)
                filename = f'{directory}All_models_pearson_heatmaps.pdf'
                filename = filename.replace(' ', '_')
                plt.savefig(filename, bbox_inches='tight', format='pdf')
                print(f'Saving to: {directory}All_models_pearson_heatmaps.pdf')
                plt.close()

        # Calculate the mean across all paths
        for title in overall_pearson_dict:
            overall_pearson_dict[title] /= path_count

        fig_size = (self.page_width, self.page_height * 0.6)  # Conversion to inches

        fig, axes = plt.subplots(2, 4, figsize=fig_size)  # Changed to 4x2 grid
        fig.suptitle('Pearson coefficient')
        axes = axes.flatten()

        wrapper = textwrap.TextWrapper(width=15, break_long_words=False, break_on_hyphens=False)
        for idx, title in enumerate(model_titles):
            ax = axes[idx]
            im = ax.imshow(overall_pearson_dict[title], cmap='viridis', origin='upper', vmin=0.95, vmax=1)
            wrapped_title = wrapper.fill(text=title)
            ax.set_title(wrapped_title)
            ax.set_xticks(range(5))
            ax.set_xticklabels(range(1, 6))
            ax.set_yticks(range(13))
            ax.set_yticklabels(range(13, 0, -1))
            ax.set_xlabel('Channel')
            ax.set_ylabel('Channel')

        for i in range(len(model_titles), len(axes)):
            fig.delaxes(axes[i])

        # Use tight_layout first to avoid overlaps
        plt.tight_layout(rect=[0, 0, 0.9, 0.99])

        # Add a color bar
        cbar_ax = fig.add_axes([0.92, 0.06, 0.02, 0.8])
        fig.colorbar(im, cax=cbar_ax)

        directory = f'plot_pearsonr_heatmap/summary/'
        self._create_directory(directory)
        filename = f'{directory}All_models_pearson_heatmaps_all_paths.pdf'
        filename = filename.replace(' ', '_')
        plt.savefig(filename, bbox_inches='tight', format='pdf')
        print(f'Saving to: {directory}All_models_pearson_heatmaps_all_paths.pdf')
        plt.close()

    def plot_normalized_correlations(self):
        """
        Plot the normalized cross-correlation of the original signal with other signals.
        """
        print("Start plotting correlations")

        wrapper = textwrap.TextWrapper(width=15, break_long_words=False, break_on_hyphens=False)

        # Storage for mean and std calculations
        all_normalized_correlations = {}
        results = []
        overall_mean_corr = {}
        overall_std_corr = {}
        path_count = 0

        # Custom colors
        mean_color = (47 / 255, 88 / 255, 110 / 255)  # RGB (47, 88, 110)
        std_color = (84 / 255, 156 / 255, 188 / 255)  # RGB (198, 222, 233)

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            path_count += 1
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original'][:, :, :60736]  # Cut to [:,:,:60736]

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
                    emg_data = emg_data[:, :, :60736]  # Cut to [:,:,:60736]
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
                    shifts = np.arange(-len(mean_corr) // 2 + 1, len(mean_corr) // 2 + 1) / 2048.0

                    # Replace NaNs with 0
                    mean_corr = np.nan_to_num(mean_corr)
                    std_corr = np.nan_to_num(std_corr)

                    # Add to overall mean and std calculations
                    if title not in overall_mean_corr:
                        overall_mean_corr[title] = mean_corr
                        overall_std_corr[title] = std_corr
                    else:
                        overall_mean_corr[title] += mean_corr
                        overall_std_corr[title] += std_corr ** 2

                    fig, ax = plt.subplots(figsize=(self.page_width, 10 / 2.54))  # Convert cm to inches
                    line1 = ax.plot(shifts, mean_corr, label='Mean', color=mean_color)
                    line2 = ax.fill_between(shifts, mean_corr - std_corr, mean_corr + std_corr, color=std_color,
                                            alpha=0.8, label='STD')

                    # Save the peak of the mean value plot
                    peak_index = int(np.argmax(mean_corr))
                    peak_mean = float(mean_corr[peak_index])
                    peak_std = float(std_corr[peak_index])
                    results.append({
                        'model': title,
                        'peak_index': peak_index,
                        'peak_mean': peak_mean,
                        'peak_std': peak_std
                    })

                    ax.set_xlabel('Shift in s')
                    ax.set_ylabel('Normalized cross-correlation')
                    ax.set_title(f'{title}')
                    ax.legend(loc='best')
                    ax.grid(True)
                    ax.set_ylim(-0.4, 1.1)
                    ax.set_xlim(-35, 35)
                    plt.tight_layout()

                    directory = f'plot_normalized_correlations/{title}/{path}/'
                    self._create_directory(directory)
                    filename = f'{directory}{title}_Mean_Normalized_Correlations.pdf'
                    filename = filename.replace(' ', '_')
                    plt.savefig(filename, format='pdf')
                    print(f'Saving summary plot for {title} - {path} to: {filename}')
                    plt.close()

        # Combine all mean and std plots into one figure for each path
        for data_dict in self.emg_data_dicts:
            for path in data_dict.keys():
                fig, axes = plt.subplots(3, 3, figsize=(self.page_width, self.page_height * 0.75))
                fig.suptitle(f'Normalized cross-correlation')
                axes = axes.flatten()

                handles_list = []
                labels_list = []

                for idx, (title, paths) in enumerate(all_normalized_correlations.items()):
                    if idx >= 9:
                        break
                    ax = axes[idx]
                    if path in paths:
                        correlations = paths[path]
                        if len(correlations) > 0:
                            correlations_array = np.array(correlations)
                            mean_corr = np.nanmean(correlations_array, axis=0)
                            std_corr = np.nanstd(correlations_array, axis=0)
                            shifts = np.arange(-len(mean_corr) // 2 + 1, len(mean_corr) // 2 + 1) / 2048.0

                            mean_corr = np.nan_to_num(mean_corr)
                            std_corr = np.nan_to_num(std_corr)

                            line1, = ax.plot(shifts, mean_corr, label='Mean', color=mean_color)
                            line2 = ax.fill_between(shifts, mean_corr - std_corr, mean_corr + std_corr, color=std_color,
                                                    alpha=0.2, label='STD')
                            ax.set_title(title)
                            ax.set_xlabel('Shift in s')
                            ax.set_ylabel('Correlation')
                            ax.grid(True)
                            ax.set_ylim(-0.4, 1.1)
                            ax.set_xlim(-35, 35)

                            if idx == 0:
                                handles_list.extend([line1, line2])
                                labels_list.extend(['Mean', 'STD'])

                # Add the legend in the subplot at position 8 (row 3, column 3)
                legend_ax = axes[8]
                legend_ax.axis('off')  # Turn off the axis
                legend_ax.legend(handles=handles_list, labels=labels_list, loc='center')

                # Hide the remaining empty subplots
                for i in range(idx + 1, 9):
                    axes[i].axis('off')

                plt.tight_layout(rect=[0, 0, 1, 1])

                directory = f'plot_normalized_correlations/summary/{path}/'
                self._create_directory(directory)
                filename = f'{directory}Combined_Mean_Normalized_Correlations_{path}.pdf'
                filename = filename.replace(' ', '_')
                fig.savefig(filename, format='pdf')
                print(f'Saving combined summary plot to: {filename}')
                plt.close(fig)

        # Calculate the overall mean and std deviation across all paths
        for title in overall_mean_corr:
            overall_mean_corr[title] /= path_count
            overall_std_corr[title] = np.sqrt(overall_std_corr[title] / path_count)

        # Plot overall mean and std deviation for all paths
        fig, axes = plt.subplots(3, 3, figsize=(self.page_width, self.page_height * 0.75))
        fig.suptitle('Normalized cross-correlation')
        axes = axes.flatten()

        handles_list = []
        labels_list = []

        for idx, title in enumerate(overall_mean_corr.keys()):
            if idx >= 9:
                break
            ax = axes[idx]
            mean_corr = overall_mean_corr[title]
            std_corr = overall_std_corr[title]
            shifts = np.arange(-len(mean_corr) // 2 + 1, len(mean_corr) // 2 + 1) / 2048.0

            line1, = ax.plot(shifts, mean_corr, label='Mean', color=mean_color)
            line2 = ax.fill_between(shifts, mean_corr - std_corr, mean_corr + std_corr, color=std_color, alpha=0.2,
                                    label='STD')
            ax.set_title(title)
            ax.set_xlabel('Shift in s')
            ax.set_ylabel('Correlation')
            ax.grid(True)
            ax.set_ylim(-0.4, 1.1)
            ax.set_xlim(-35, 35)

            # Save the peak of the combined mean value plot
            peak_index = int(np.argmax(mean_corr))
            peak_mean = float(mean_corr[peak_index])
            peak_std = float(std_corr[peak_index])
            results.append({
                'model': f'Combined_{title}',
                'peak_index': peak_index,
                'peak_mean': peak_mean,
                'peak_std': peak_std
            })

            if idx == 0:
                handles_list.extend([line1, line2])
                labels_list.extend(['Mean', 'STD'])

        # Add the legend in the subplot at position 8 (row 3, column 3)
        legend_ax = axes[8]
        legend_ax.axis('off')  # Turn off the axis
        legend_ax.legend(handles=handles_list, labels=labels_list, loc='center')

        # Hide the remaining empty subplots
        for i in range(idx + 1, 9):
            axes[i].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 1])

        directory = f'plot_normalized_correlations/summary/'
        self._create_directory(directory)
        filename = f'{directory}Combined_Mean_Normalized_Correlations_All_Paths.pdf'
        filename = filename.replace(' ', '_')
        fig.savefig(filename, format='pdf')
        print(f'Saving combined summary plot to: {filename}')
        plt.close(fig)

        # Save the results to a CSV file
        results_path = 'plot_normalized_correlations/summary/metrics.csv'
        self._create_directory('plot_normalized_correlations/summary/')
        with open(results_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['model', 'peak_index', 'peak_mean', 'peak_std'])
            writer.writeheader()
            writer.writerows(results)
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
                    fig, axes = plt.subplots(nrows=13, ncols=5, figsize=(self.page_width * 1.5, self.page_height))
                    fig.suptitle(f'Error: Original vs {title}')

                    # Additional figure for the large plot at position [1][4]
                    fig_large, ax_large = plt.subplots(figsize=(self.page_width, self.page_height))  # A4 page in inches

                    for i in range(13):
                        for j in range(5):
                            ax = axes[i, j]
                            time_array = np.arange(diff.shape[2]) / 2048.0
                            ax.plot(time_array, diff[i, j], color=plot_color, linewidth=0.5)
                            if j == 0:
                                ax.set_ylabel('Error\nin µV')
                            ax.set_ylim(global_min, global_max)
                            if i != 12:
                                ax.set_xticklabels([])
                            if j != 0:
                                ax.set_yticklabels([])
                            if i == 12:
                                ax.set_xlabel('Time in s')
                            if i == 1 and j == 4:
                                ax_large.plot(time_array, diff[i, j], color=plot_color, linewidth=0.5)
                                ax_large.set_title(f'Error at Row 2, Column 5 - {path}')
                                ax_large.set_xlabel('Time in s')
                                ax_large.set_ylabel('Error in µV')
                                ax_large.set_ylim(global_min, global_max)

                            ax.grid(True, which='both')

                            # Set minor ticks
                            ax.yaxis.set_minor_locator(MultipleLocator(250))
                            ax.xaxis.set_major_locator(MultipleLocator(10))

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

                # Create a single figure for both positions with subplots for all models
                fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(self.page_width, self.page_height))
                axes = axes.flatten()

                for idx, (title, grid_2) in enumerate(
                        {k: v for k, v in datasets.items() if k != 'Original'}.items()):
                    if idx >= 14:
                        break  # Ensure we don't create more than 14 subplots (7 rows x 2 columns)

                    diff = groundtruth - grid_2

                    # Given channel at position (4, 2)
                    row1, col1 = 4, 2
                    ax1 = axes[2 * idx]  # Position in the first column
                    time_array1 = np.arange(diff.shape[2]) / 2048.0
                    ax1.plot(time_array1, diff[row1, col1], color=plot_color, linewidth=0.5)
                    ax1.set_title(f'{title}')
                    ax1.set_ylim(global_min, global_max)
                    ax1.grid(True, which='both')

                    # Set minor ticks
                    ax1.yaxis.set_minor_locator(MultipleLocator(250))

                    if idx == 6:  # Add x-axis label only to the bottom plots
                        ax1.set_xlabel('Time in s')
                    else:
                        ax1.set_xticklabels([])
                    ax1.set_ylabel('Error in µV')  # Add y-axis label only to the first column

                    # Predicted channel at position (3, 1)
                    row2, col2 = 3, 1
                    ax2 = axes[2 * idx + 1]  # Position in the second column
                    time_array2 = np.arange(diff.shape[2]) / 2048.0
                    line2, = ax2.plot(time_array2, diff[row2, col2], color=plot_color, linewidth=0.5)
                    ax2.set_title(f'{title}')
                    ax2.set_ylim(global_min, global_max)
                    ax2.grid(True, which='both')

                    # Set minor ticks
                    ax2.yaxis.set_minor_locator(MultipleLocator(250))

                    if idx == 6:  # Add x-axis label only to the bottom plots
                        ax2.set_xlabel('Time in s')
                    else:
                        ax2.set_xticklabels([])
                    ax2.set_yticklabels([])

                # Add column titles
                fig.text(0.31, 0.95, 'Error - Given channel', ha='center', fontsize=10)
                fig.text(0.78, 0.95, 'Error - Predicted channel', ha='center', fontsize=10)

                fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for column titles

                # Save the combined figure for both positions
                directory = f'plot_difference/{path}/positions/'
                self._create_directory(directory)
                filename = f'{directory}Differences_Positions_Combined.pdf'
                filename = filename.replace(' ', '_')
                plt.savefig(filename, format='pdf')
                print(f'Saving to: {filename}')
                plt.close(fig)

                # Create a single figure differentiating given and predicted channels with subplots for all models
                fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(self.page_width, self.page_height))
                axes = axes.flatten()

                for idx, (title, grid_2) in enumerate(
                        {k: v for k, v in datasets.items() if k != 'Original'}.items()):
                    if idx >= 14:
                        break  # Ensure we don't create more than 14 subplots (7 rows x 2 columns)

                    diff = groundtruth - grid_2

                    # Given channels
                    given_diff = diff[::2, ::2]
                    mean_given_diff = np.mean(given_diff, axis=(0, 1))
                    local_min_given = np.floor(np.min(mean_given_diff) / 200) * 200
                    local_max_given = np.ceil(np.max(mean_given_diff) / 200) * 200
                    ax1 = axes[2 * idx]  # Position in the first column
                    time_array1 = np.arange(mean_given_diff.shape[0]) / 2048.0
                    ax1.plot(time_array1, mean_given_diff, color=plot_color, linewidth=0.5)
                    ax1.set_title(f'{title}')
                    ax1.set_ylim(local_min_given, local_max_given)
                    ax1.grid(True, which='both')
                    ax1.yaxis.set_minor_locator(MultipleLocator(250))
                    if idx == 6:  # Add x-axis label only to the bottom plots
                        ax1.set_xlabel('Time in s')
                    else:
                        ax1.set_xticklabels([])
                    ax1.set_ylabel('Error in µV')  # Add y-axis label only to the first column

                    # Create a mask for given channels
                    mask_given = np.zeros_like(diff, dtype=bool)
                    mask_given[::2, ::2, :] = True

                    # Invert the mask to get predicted channels
                    mask_predicted = ~mask_given

                    # Extract predicted channels using the mask
                    predicted_diff = diff[mask_predicted].reshape(-1, diff.shape[2])
                    mean_predicted_diff = np.mean(predicted_diff, axis=0)
                    local_min_predicted = np.floor(np.min(mean_predicted_diff) / 200) * 200
                    local_max_predicted = np.ceil(np.max(mean_predicted_diff) / 200) * 200
                    ax2 = axes[2 * idx + 1]  # Position in the second column
                    time_array2 = np.arange(mean_predicted_diff.shape[0]) / 2048.0
                    ax2.plot(time_array2, mean_predicted_diff, color=plot_color, linewidth=0.5)
                    ax2.set_title(f'{title}')
                    ax2.set_ylim(local_min_predicted, local_max_predicted)
                    ax2.grid(True, which='both')
                    ax2.yaxis.set_minor_locator(MultipleLocator(250))
                    if idx == 6:  # Add x-axis label only to the bottom plots
                        ax2.set_xlabel('Time in s')
                    else:
                        ax2.set_xticklabels([])

                # Add column titles
                fig.text(0.31, 0.95, 'Error - Given channels', ha='center', fontsize=10)
                fig.text(0.78, 0.95, 'Error - Predicted channels', ha='center', fontsize=10)

                fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for column titles

                # Save the combined figure for given and predicted channels
                directory = f'plot_difference/{path}/positions/'
                self._create_directory(directory)
                filename = f'{directory}Differences_Given_Predicted_Combined.pdf'
                filename = filename.replace(' ', '_')
                plt.savefig(filename, format='pdf')
                print(f'Saving to: {filename}')
                plt.close(fig)

        print("Finished plotting all differences.")

    def propagation_plot(self) -> None:
        """
        Creates a figure with 13 subplots for each element in the grids dictionary.
        Each subplot shows the signal [y][2][start_idx:end_idx] for the third column of the array.
        """
        print("Start plotting propagation plots")

        start_idx = 30050
        end_idx = start_idx + 256

        # Calculate the time array in milliseconds
        time_array = (np.arange(start_idx, end_idx) - start_idx) / 2048.0 * 1000  # in milliseconds

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                # Iterate over datasets
                for title, array in datasets.items():
                    fig, axes = plt.subplots(nrows=13, ncols=1, figsize=(self.page_width, self.page_height))
                    fig.suptitle(f'Propagation Plot: {title}')

                    for y in range(13):
                        ax = axes[y]
                        ax.plot(time_array, array[y, 2, start_idx:end_idx])  # Adjusted to plot from the third dimension
                        if y == 12:  # Only set x-axis labels on the last subplot
                            ax.set_xlabel('Time in ms')
                        else:
                            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                        ax.set_yticks([])

                    plt.tight_layout()
                    plt.subplots_adjust(top=0.95)
                    directory = f'plot_propagation/{path}/'
                    self._create_directory(directory)
                    filename = f'{directory}Propagation_Plot_{title}.pdf'
                    filename = filename.replace(' ', '_')
                    plt.savefig(filename, format='pdf')
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

        # Define figure size for the main plots
        fig_size = (self.page_width, self.page_height)  # Increased height to 33.7 cm and converted to inches

        # Storage for overall mean and std correlations
        overall_mean_corr_predicted = {}
        overall_mean_corr_given = {}
        overall_std_corr_predicted = {}
        overall_std_corr_given = {}
        path_count = 0

        # Custom colors
        mean_color = (47 / 255, 88 / 255, 110 / 255)  # RGB (47, 88, 110)
        std_color = (84 / 255, 156 / 255, 188 / 255)  # RGB (84, 156, 188)

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            path_count += 1
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original'][:, :, :60736]
                linear_interpolation = datasets['Bilinear'][:, :, :60736]

                # Iterate over datasets, skipping the 'Original' itself
                for title, emg_data in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    emg_data = emg_data[:, :, :60736]
                    directory = f'plot_windowed_correlations/{path}/{title}/'
                    self._create_directory(directory)

                    # Plot differences if the current title is not 'Bilinear interpolation'
                    if title != 'Bilinear':
                        diff_directory = f'plot_windowed_correlations/{path}/Linear_vs_{title}/'
                        self._create_directory(diff_directory)

                        # Determine global min and max for differences across all subplots
                        global_diff_y_min, global_diff_y_max = float('inf'), float('-inf')
                        global_diff_x_min, global_diff_x_max = float('inf'), float('-inf')

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

                                # Update global min and max for differences
                                global_diff_y_min = min(global_diff_y_min, min(differences))
                                global_diff_y_max = max(global_diff_y_max, max(differences))
                                global_diff_x_min = min(global_diff_x_min, min(x_axis_diff))
                                global_diff_x_max = max(global_diff_x_max, max(x_axis_diff))

                                # Add to big subplot figure for differences
                                ax_diff = axes_diff[row, col]
                                ax_diff.fill_between(x_axis_diff, differences, where=(differences >= 0),
                                                     color='lightgreen', alpha=0.5)
                                ax_diff.fill_between(x_axis_diff, differences, where=(differences < 0),
                                                     color='lightcoral', alpha=0.5)
                                ax_diff.plot(x_axis_diff, differences, color='black')
                                ax_diff.set_ylim(global_diff_y_min, global_diff_y_max)
                                ax_diff.set_xlim(global_diff_x_min, global_diff_x_max)
                                if row == 12:  # Add x-axis label only to the bottom plots
                                    ax_diff.set_xlabel('Index')
                                if col == 0:  # Add y-axis label only to the first column
                                    ax_diff.set_ylabel('Diff.')
                                ax_diff.label_outer()

                        # Save the large difference figure
                        fig_diff.tight_layout(rect=[0, 0, 1, 0.95])
                        fig_diff.suptitle(
                            f'Difference in Windowed Pearson Correlation\nBilinear interpolation vs {title}', y=1.02)
                        filename = f'{diff_directory}Combined_Difference_{title}.pdf'
                        filename = filename.replace(' ', '_')
                        fig_diff.savefig(filename)
                        print(f'Saving to: {filename}')
                        plt.close(fig_diff)

                        # Additional plots for positions [3][1] and [4][2]
                        positions = [(3, 1), (4, 2)]
                        fig_pred, axes_pred = plt.subplots(2, 3, figsize=(self.page_width, self.page_height * 0.4))  # 2 rows and 3 columns for predicted position
                        fig_given, axes_given = plt.subplots(2, 3, figsize=(self.page_width, self.page_height * 0.4))  # 2 rows and 3 columns for given position

                        # Flatten axes for easier indexing
                        axes_pred = axes_pred.flatten()
                        axes_given = axes_given.flatten()

                        # Dictionary to hold figure and axes references for easier looping
                        fig_axes = {
                            "predicted": (fig_pred, axes_pred),
                            "given": (fig_given, axes_given)
                        }

                        for pos_idx, (row, col) in enumerate(positions):
                            fig, axes = fig_axes["predicted" if pos_idx == 0 else "given"]

                            for i, other_title in enumerate([k for k in datasets.keys() if k != 'Bilinear' and k != 'Original']):
                                signal_linear = linear_interpolation[row, col, :]
                                signal_compare = datasets[other_title][row, col, :]
                                correlations_linear = self.calculate_windowed_correlation(groundtruth[row, col, :], signal_linear, window_size)
                                correlations_compare = self.calculate_windowed_correlation(groundtruth[row, col, :], signal_compare, window_size)
                                differences = np.array(correlations_compare) - np.array(correlations_linear)
                                x_axis_diff = np.arange(len(differences))

                                ax = axes[i]
                                ax.fill_between(x_axis_diff, differences, where=(differences >= 0), color='lightgreen', alpha=0.5)
                                ax.fill_between(x_axis_diff, differences, where=(differences < 0), color='lightcoral', alpha=0.5)
                                ax.plot(x_axis_diff, differences, color='black')
                                ax.set_title(f'{other_title}')
                                if i % 3 == 0:  # Add y-axis label only to the first column
                                    ax.set_ylabel('Difference')
                                if i >= 3:  # Add x-axis label only to the second row
                                    ax.set_xlabel('Index')
                                ax.set_ylim(-0.2, 0.2)
                                ax.set_xlim(0, 120)

                        fig_pred.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to make space for column titles
                        fig_pred.suptitle(f'Difference in windowed Pearson correlation\nBilinear interpolation vs other models - Predicted channel')
                        fig_given.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust layout to make space for column titles
                        fig_given.suptitle(f'Difference in windowed Pearson correlation\nBilinear interpolation vs other models - Given channel')

                        filename_pred = f'{diff_directory}Combined_Difference_Predicted_Position.pdf'
                        filename_pred = filename_pred.replace(' ', '_')
                        fig_pred.savefig(filename_pred)
                        print(f'Saving to: {filename_pred}')

                        filename_given = f'{diff_directory}Combined_Difference_Given_Position.pdf'
                        filename_given = filename_given.replace(' ', '_')
                        fig_given.savefig(filename_given)
                        print(f'Saving to: {filename_given}')

                        plt.close(fig_pred)
                        plt.close(fig_given)

        # Summarize all paths into one figure
        for data_dict in self.emg_data_dicts:
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original'][:, :, :60736]
                linear_interpolation = datasets['Bilinear'][:, :, :60736]

                for title, emg_data in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    emg_data = emg_data[:, :, :60736]

                    for row in range(13):
                        for col in range(5):
                            signal_linear = linear_interpolation[row, col, :]
                            signal_compare = emg_data[row, col, :]
                            correlations_linear = self.calculate_windowed_correlation(groundtruth[row, col, :], signal_linear, window_size)
                            correlations_compare = self.calculate_windowed_correlation(groundtruth[row, col, :], signal_compare, window_size)
                            differences = np.array(correlations_compare) - np.array(correlations_linear)

                            if (row, col) in [(3, 1), (4, 2)]:
                                key = 'predicted' if (row, col) == (3, 1) else 'given'
                                if title not in overall_mean_corr_predicted:
                                    overall_mean_corr_predicted[title] = []
                                    overall_std_corr_predicted[title] = []
                                    overall_mean_corr_given[title] = []
                                    overall_std_corr_given[title] = []

                                if key == 'predicted':
                                    overall_mean_corr_predicted[title].append(differences)
                                else:
                                    overall_mean_corr_given[title].append(differences)

        # Calculate mean and std for overall correlations
        for title in overall_mean_corr_predicted.keys():
            overall_mean_corr_predicted[title] = np.nanmean(np.vstack(overall_mean_corr_predicted[title]), axis=0)
            overall_std_corr_predicted[title] = np.nanstd(np.vstack(overall_mean_corr_predicted[title]), axis=0)
            overall_mean_corr_given[title] = np.nanmean(np.vstack(overall_mean_corr_given[title]), axis=0)
            overall_std_corr_given[title] = np.nanstd(np.vstack(overall_mean_corr_given[title]), axis=0)

        # Plot overall mean and std deviation for predicted channels
        fig, axes = plt.subplots(2, 3, figsize=(self.page_width, self.page_height * 0.4))
        fig.suptitle('Overall Windowed Pearson correlation - Predicted channels')
        axes = axes.flatten()

        for idx, title in enumerate(overall_mean_corr_predicted.keys()):
            if idx >= 6:
                break
            ax = axes[idx]
            mean_corr = overall_mean_corr_predicted[title]
            std_corr = overall_std_corr_predicted[title]
            shifts = np.arange(-len(mean_corr) // 2 + 1, len(mean_corr) // 2 + 1) / 2048.0

            ax.plot(shifts, mean_corr, label='Mean', color=mean_color)
            ax.fill_between(shifts, mean_corr - std_corr, mean_corr + std_corr, color=std_color, alpha=0.2, label='STD')
            ax.set_title(f'{title}')
            ax.set_xlabel('Shift in s')
            ax.set_ylabel('Correlation')
            ax.grid(True)
            ax.set_ylim(-0.2, 0.2)
            ax.set_xlim(-35, 35)

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        directory = f'plot_windowed_correlations/summary/'
        self._create_directory(directory)
        filename = f'{directory}Overall_Windowed_Correlation_Predicted_Channels.pdf'
        filename = filename.replace(' ', '_')
        fig.savefig(filename)
        print(f'Saving to: {filename}')
        plt.close(fig)

        # Plot overall mean and std deviation for given channels
        fig, axes = plt.subplots(2, 3, figsize=(self.page_width, self.page_height * 0.4))
        fig.suptitle('Overall Windowed Pearson correlation - Given channels')
        axes = axes.flatten()

        for idx, title in enumerate(overall_mean_corr_given.keys()):
            if idx >= 6:
                break
            ax = axes[idx]
            mean_corr = overall_mean_corr_given[title]
            std_corr = overall_std_corr_given[title]
            shifts = np.arange(-len(mean_corr) // 2 + 1, len(mean_corr) // 2 + 1) / 2048.0

            ax.plot(shifts, mean_corr, label='Mean', color=mean_color)
            ax.fill_between(shifts, mean_corr - std_corr, mean_corr + std_corr, color=std_color, alpha=0.2, label='STD')
            ax.set_title(f'{title}')
            ax.set_xlabel('Shift in s')
            ax.set_ylabel('Correlation')
            ax.grid(True)
            ax.set_ylim(-0.2, 0.2)
            ax.set_xlim(-35, 35)

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        directory = f'plot_windowed_correlations/summary/'
        self._create_directory(directory)
        filename = f'{directory}Overall_Windowed_Correlation_Given_Channels.pdf'
        filename = filename.replace(' ', '_')
        fig.savefig(filename)
        print(f'Saving to: {filename}')
        plt.close(fig)

        print("Finished plotting all correlations and differences.")

    def create_box_plot(self):
        data_dicts = self._load_data()
        self._create_directory('boxplots')

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
                                 keep_outliers=False, group_spacing=2, box_spacing=0, box_width=0.7):
            boxplot_data = []
            labels = []  # Will hold labels for the x-axis, which are model names wrapped if needed

            # Creating a TextWrapper object to wrap text without breaking words
            wrapper = textwrap.TextWrapper(width=15, break_long_words=False, break_on_hyphens=False)

            for model in differences_all.keys():
                all_diffs = differences_all[model]
                even_diffs = differences_even[model]
                odd_diffs = differences_odd[model]

                boxplot_data.append(all_diffs)
                boxplot_data.append(even_diffs)
                boxplot_data.append(odd_diffs)
                labels.append(wrapper.fill(text=model))

            fig, ax = plt.subplots(figsize=(self.page_width , self.page_height*0.4))
            positions = []
            for i in range(len(differences_all)):
                base_position = i * (3 * box_width + group_spacing)
                positions.extend([base_position + j * (box_width + box_spacing) for j in range(3)])

            # Set showfliers to the opposite of remove_outliers
            bp = ax.boxplot(boxplot_data, positions=positions, patch_artist=True, widths=box_width,
                            showfliers=keep_outliers)

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
                plt.Line2D([0], [0], color=colors[1], lw=4, label='Given\nchannels'),
                plt.Line2D([0], [0], color=colors[2], lw=4, label='Predicted\nchannels'),
            ]

            # Manually set the y-axis limit if removing outliers
            if keep_outliers:
                y_max = max([max(data) for data in boxplot_data if data]) * 1.04 # Increase by 6% for space

            else:
                y_max = max(
                    [np.percentile(data, 75) + 1.5 * (np.percentile(data, 75) - np.percentile(data, 25)) for data in
                     boxplot_data if data]) * 1.04

            ax.set_ylim(0, y_max)

            # Setting title with bold font and adjusting spacing
            ax.set_title(title)

            plt.xlabel('Models')
            plt.ylabel('Absolute error in µV')
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

        # Initialize dictionaries to hold differences for each model across all paths for each subset
        differences_all = {model: [] for model in [
            'Bilinear', 'Bicubic spline', 'SRGAN', 'UNet', 'UNet Fourier',
            'UNet Wavelet', 'CNN-LSTM'
        ]}
        differences_even = {model: [] for model in differences_all}
        differences_odd = {model: [] for model in differences_all}

        # Calculate the absolute differences
        for data_dict in data_dicts:
            for path, data in data_dict.items():
                path_differences_all = {model: [] for model in differences_all}
                path_differences_even = {model: [] for model in differences_even}
                path_differences_odd = {model: [] for model in differences_odd}

                origin_hr = data['Original']
                for model, model_data in data.items():
                    if model != 'Original':
                        # All data
                        all_diffs = np.abs(model_data - origin_hr).flatten()
                        path_differences_all[model].extend(all_diffs)
                        differences_all[model].extend(all_diffs)

                        # Even rows and columns (0,0), (0,2), (2,0), etc.
                        even_diffs = np.abs(model_data[::2, ::2] - origin_hr[::2, ::2]).flatten()
                        path_differences_even[model].extend(even_diffs)
                        differences_even[model].extend(even_diffs)

                        # Odd columns completely and odd rows in the even columns
                        odd_diffs_cols_completely = np.abs(model_data[:, 1::2] - origin_hr[:, 1::2]).flatten()
                        odd_diffs_rows_even_cols = np.abs(model_data[1::2, ::2] - origin_hr[1::2, ::2]).flatten()

                        path_differences_odd[model].extend(odd_diffs_cols_completely)
                        path_differences_odd[model].extend(odd_diffs_rows_even_cols)
                        differences_odd[model].extend(odd_diffs_cols_completely)
                        differences_odd[model].extend(odd_diffs_rows_even_cols)

                # Plot grouped boxplot without outliers for each path
                plot_grouped_boxplot(path_differences_all, path_differences_even, path_differences_odd,
                                     f'Absolute error between real and synthetic data (Without outliers)',
                                     colors,
                                     f'boxplots/boxplot_grouped_data_no_outliers_{path}.png', keep_outliers=False)

                # Plot grouped boxplot with outliers for each path
                plot_grouped_boxplot(path_differences_all, path_differences_even, path_differences_odd,
                                     f'Absolute error between real and synthetic data (With outliers)',
                                     colors,
                                     f'boxplots/boxplot_grouped_data_with_outliers_{path}.png', keep_outliers=True)

        # Plot grouped boxplot without outliers for all data
        plot_grouped_boxplot(differences_all, differences_even, differences_odd,
                             'Absolute error between real and synthetic data (Without outliers)',
                             colors,
                             'boxplots/boxplot_grouped_data_no_outliers.png', keep_outliers=False)

        # Plot grouped boxplot with outliers for all data
        plot_grouped_boxplot(differences_all, differences_even, differences_odd,
                             'Absolute error between real and synthetic data (With outliers)', colors,
                             'boxplots/boxplot_grouped_data_with_outliers.png', keep_outliers=True)

        # Save metrics for all data
        save_metrics(differences_all, 'boxplots/metrics_all_data.csv')
        save_metrics(differences_even, 'boxplots/metrics_given_channels.csv')
        save_metrics(differences_odd, 'boxplots/metrics_predicted_channels.csv')

        print("Finished creating box plots.")

    def plot_specific_positions(self):
        """
        Create plots for specific positions [3][1] (predicted channel) and [4][2] (given channel) for each path and selected models.
        """
        given_position = (4, 2)
        predicted_position = (3, 1)
        x_range = slice(30000, 30128)  # x-axis range from 30000 to 30128 (128 samples)

        models = [
            'Bilinear', 'Bicubic spline', 'SRGAN',
            'UNet', 'UNet Fourier', 'UNet Wavelet', 'CNN-LSTM', 'Original'
        ]

        # Define colors matching the order of the models
        colors = [
            (127 / 255, 127 / 255, 127 / 255), (47 / 255, 88 / 255, 110 / 255),
            (255 / 255, 192 / 255, 0 / 255), (239 / 255, 167 / 255, 141 / 255),
            (0 / 255, 176 / 255, 240 / 255), (112 / 255, 48 / 255, 160 / 255),
            (0 / 255, 128 / 255, 0 / 255), (148 / 255, 11 / 255, 45 / 255)
        ]

        # Calculate the time array in milliseconds
        time_array = (np.arange(x_range.start, x_range.stop) - x_range.start) / 2048.0 * 1000  # in milliseconds

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                # Create a figure with two subplots
                fig, axs = plt.subplots(1, 2, figsize=(
                self.page_width, self.page_height * 0.5))  # Increase width to 24 cm and height to 14 cm

                # Plot for the given channel in the first column
                ax_given = axs[0]
                for color, title in zip(colors[:-1], models[:-1]):
                    if title in datasets:
                        alpha = 0.7
                        signal = datasets[title][given_position[0], given_position[1], x_range]
                        ax_given.plot(time_array, signal, label=title, alpha=alpha, color=color, linewidth=1)

                if 'Original' in datasets:
                    color = colors[-1]
                    title = 'Original'
                    alpha = 0.7
                    signal = datasets[title][given_position[0], given_position[1], x_range]
                    ax_given.plot(time_array, signal, label=title, alpha=alpha, color=color, linewidth=1)

                ax_given.set_xlabel('Time in ms')
                ax_given.set_ylabel('Amplitude in µV')
                ax_given.set_title('sEMG - Given channel')

                # Plot for the predicted channel in the second column
                ax_predicted = axs[1]
                for color, title in zip(colors[:-1], models[:-1]):
                    if title in datasets:
                        alpha = 0.7
                        signal = datasets[title][predicted_position[0], predicted_position[1], x_range]
                        ax_predicted.plot(time_array, signal, label=title, alpha=alpha, color=color, linewidth=1)

                if 'Original' in datasets:
                    color = colors[-1]
                    title = 'Original'
                    alpha = 0.7
                    signal = datasets[title][predicted_position[0], predicted_position[1], x_range]
                    ax_predicted.plot(time_array, signal, label=title, alpha=alpha, color=color, linewidth=1)

                ax_predicted.set_xlabel('Time in ms')
                ax_predicted.set_ylabel('Amplitude in µV')
                ax_predicted.set_title('sEMG - Predicted channel')

                # Add a single legend in the middle below the two subplots
                handles, labels = axs[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4)

                plt.tight_layout()

                # Create directory and save the plot
                directory = f'plot_specific_positions/{path}/'
                self._create_directory(directory)
                filename = f'{directory}Position_3_1_and_4_2.pdf'
                filename = filename.replace(' ', '_')
                plt.savefig(filename, bbox_inches='tight', format='pdf')
                plt.close()

                print(f'Saving to: {directory}Position_3_1_and_4_2.pdf')

        print("Finished plotting specific positions.")

    # def plot_frequency_spectrum_difference(self):
    #     """
    #     Plot the differences in the frequency spectrum between the groundtruth signal and other signals.
    #     """
    #     print("Start plotting frequency spectrum differences")
    #
    #     # Function to compute the Fourier Transform and frequency spectrum
    #     def compute_frequency_spectrum(signal, fs=2048):
    #         n = len(signal)
    #         freq = np.fft.fftfreq(n, d=1 / fs)
    #         spectrum = np.fft.fft(signal)
    #         return freq, np.abs(spectrum)
    #
    #     positions = [(3, 1), (4, 2)]  # Only these two positions
    #
    #     # Custom colors
    #     spectrum_color = (47 / 255, 88 / 255, 110 / 255)  # RGB (47, 88, 110)
    #     difference_color = (199 / 255, 91 / 255, 76 / 255)
    #
    #     # Storage for overall mean and std correlations
    #     overall_spectrum_gt = {}
    #     overall_spectrum_diff = {}
    #     path_count = 0
    #
    #     # Iterate over each dictionary in the list
    #     for data_dict in self.emg_data_dicts:
    #         path_count += 1
    #         # Unpack the path and its datasets from each dictionary
    #         for path, datasets in data_dict.items():
    #             groundtruth = datasets['Original']
    #
    #             for pos_idx, pos in enumerate(positions):
    #                 row, col = pos
    #                 fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(self.page_width * 1.5, self.page_height * 0.75))
    #                 axes = axes.flatten()
    #
    #                 # Variables to collect handles and labels for the combined legend
    #                 handles_list = []
    #                 labels_list = []
    #
    #                 min_length = float('inf')
    #
    #                 for idx, (title, emg_data) in enumerate(
    #                         {k: v for k, v in datasets.items() if k != 'Original'}.items()):
    #                     if idx >= 7:
    #                         break  # Ensure we don't create more than 7 subplots
    #
    #                     ax = axes[idx]
    #                     groundtruth_sig = groundtruth[row, col, :]
    #                     signal2 = emg_data[row, col, :]
    #
    #                     freq_gt, spectrum_gt = compute_frequency_spectrum(groundtruth_sig)
    #                     freq_sig, spectrum_sig = compute_frequency_spectrum(signal2)
    #
    #                     spectrum_diff = spectrum_gt - spectrum_sig
    #
    #                     # Find the minimum length of the spectra
    #                     min_length = min(min_length, len(spectrum_gt), len(spectrum_diff))
    #
    #                     # Update overall spectrum data
    #                     if title not in overall_spectrum_gt:
    #                         overall_spectrum_gt[title] = []
    #                         overall_spectrum_diff[title] = []
    #
    #                     overall_spectrum_gt[title].append(spectrum_gt)
    #                     overall_spectrum_diff[title].append(spectrum_diff)
    #
    #                     line1, = ax.plot(freq_gt, spectrum_gt, label='Original spectrum', color=spectrum_color, linewidth=0.5)
    #                     line2, = ax.plot(freq_gt, spectrum_diff, label='Spectrum error', color=difference_color, linewidth=0.5)
    #                     ax.set_xlim(0, 600)  # Limiting the frequency range for better visualization
    #                     ax.set_title(f'{title}')
    #                     ax.grid(True)
    #                     ax.set_ylabel('Magnitude')
    #                     ax.set_xlabel('Frequency in Hz')
    #
    #                     # Collect handles and labels only once
    #                     if idx == 0:
    #                         handles_list.append(line1)
    #                         handles_list.append(line2)
    #                         labels_list.append('Original spectrum')
    #                         labels_list.append('Spectrum error')
    #
    #                 # Truncate all spectra to the minimum length found
    #                 for title in overall_spectrum_gt.keys():
    #                     overall_spectrum_gt[title] = [spectrum[:min_length] for spectrum in overall_spectrum_gt[title]]
    #                     overall_spectrum_diff[title] = [spectrum[:min_length] for spectrum in
    #                                                     overall_spectrum_diff[title]]
    #
    #                 # Create custom legend handles with different line widths
    #                 legend_handles = [plt.Line2D([], [], color=handle.get_color(), linewidth=1.5) for handle in
    #                                   handles_list]
    #
    #                 # Add legend to the last subplot if it's empty
    #                 legend_ax = axes[8]
    #                 legend_ax.axis('off')
    #                 legend_ax.legend(handles=legend_handles, labels=labels_list, loc='center')
    #
    #                 # Hide any remaining empty subplots
    #                 for i in range(idx + 1, 9):
    #                     axes[i].axis('off')
    #
    #                 # Set title for the figure
    #                 channel_type = "predicted" if pos_idx == 0 else "given"
    #                 fig.suptitle(f'Error {channel_type} channel')
    #                 fig.tight_layout()
    #
    #                 # Save the figure for each position
    #                 directory = f'plot_frequency_spectrum_difference/{path}/'
    #                 self._create_directory(directory)
    #                 filename = f'{directory}Frequency_Spectrum_Comparison_{channel_type}_channel.pdf'
    #                 filename = filename.replace(' ', '_')
    #                 plt.savefig(filename, bbox_inches='tight', format='pdf')
    #                 print(f'Saving to: {filename}')
    #                 plt.close(fig)

    def plot_frequency_spectrum_difference(self):
        """
        Plot the differences in the frequency spectrum between the groundtruth signal and other signals.
        """
        print("Start plotting frequency spectrum differences")

        # Function to compute the Fourier Transform and frequency spectrum
        def compute_frequency_spectrum(signal, fs=2048):
            n = len(signal)
            freq = np.fft.fftfreq(n, d=1 / fs)
            spectrum = np.abs(np.fft.fft(signal))
            return freq, spectrum

        # Custom colors
        spectrum_color = (47 / 255, 88 / 255, 110 / 255)  # RGB (47, 88, 110)
        difference_color = (199 / 255, 91 / 255, 76 / 255)

        # Iterate over each dictionary in the list
        for data_dict in self.emg_data_dicts:
            # Unpack the path and its datasets from each dictionary
            for path, datasets in data_dict.items():
                groundtruth = datasets['Original']

                # Containers for storing spectra for given and predicted channels
                spectra_given = {}
                spectra_predicted = {}
                spectra_given_positions = {}
                spectra_predicted_positions = {}

                # Process each model
                for title, emg_data in {k: v for k, v in datasets.items() if k != 'Original'}.items():
                    given_spectra_gt = []
                    given_spectra_diff = []
                    predicted_spectra_gt = []
                    predicted_spectra_diff = []

                    for row in range(13):
                        for col in range(5):
                            groundtruth_sig = groundtruth[row, col, :]
                            signal2 = emg_data[row, col, :]
                            freq_gt, spectrum_gt = compute_frequency_spectrum(groundtruth_sig)
                            freq_sig, spectrum_sig = compute_frequency_spectrum(signal2)
                            spectrum_diff = spectrum_gt - spectrum_sig

                            if (row % 2 == 0 and col % 2 == 0):  # Given channel
                                given_spectra_gt.append(spectrum_gt)
                                given_spectra_diff.append(spectrum_diff)
                            else:  # Predicted channel
                                predicted_spectra_gt.append(spectrum_gt)
                                predicted_spectra_diff.append(spectrum_diff)

                            # Specific positions
                            if (row, col) == (4, 2):
                                spectra_given_positions[title] = (spectrum_gt, spectrum_diff)
                            if (row, col) == (3, 1):
                                spectra_predicted_positions[title] = (spectrum_gt, spectrum_diff)

                    # Calculate mean spectra
                    given_spectra_gt = np.mean(given_spectra_gt, axis=0)
                    given_spectra_diff = np.mean(given_spectra_diff, axis=0)
                    predicted_spectra_gt = np.mean(predicted_spectra_gt, axis=0)
                    predicted_spectra_diff = np.mean(predicted_spectra_diff, axis=0)

                    # Store the computed mean spectra
                    spectra_given[title] = (given_spectra_gt, given_spectra_diff)
                    spectra_predicted[title] = (predicted_spectra_gt, predicted_spectra_diff)

                # Plot the spectra for given channels
                fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(self.page_width * 1.5, self.page_height * 0.75))
                axes = axes.flatten()

                handles_list = []
                labels_list = []

                for idx, title in enumerate(spectra_given.keys()):
                    if idx >= 7:
                        break  # Ensure we don't create more than 7 subplots

                    ax = axes[idx]
                    freq = np.fft.fftfreq(len(spectra_given[title][0]), d=1 / 2048.0)
                    line1, = ax.plot(freq, spectra_given[title][0], label='Original spectrum', color=spectrum_color,
                                     linewidth=0.5)
                    line2, = ax.plot(freq, spectra_given[title][1], label='Spectrum error', color=difference_color,
                                     linewidth=0.5)
                    ax.set_xlim(0, 600)  # Limiting the frequency range for better visualization
                    ax.set_title(f'{title}')
                    ax.grid(True)
                    ax.set_ylabel('Magnitude')
                    ax.set_xlabel('Frequency in Hz')

                    if idx == 0:
                        handles_list.append(line1)
                        handles_list.append(line2)
                        labels_list.append('Original spectrum')
                        labels_list.append('Spectrum error')

                legend_handles = [plt.Line2D([], [], color=handle.get_color(), linewidth=1.5) for handle in
                                  handles_list]

                legend_ax = axes[8]
                legend_ax.axis('off')
                legend_ax.legend(handles=legend_handles, labels=labels_list, loc='center')

                for i in range(idx + 1, 9):
                    axes[i].axis('off')

                fig.suptitle(f'Error - Given channels')
                fig.tight_layout()

                directory = f'plot_frequency_spectrum_difference/{path}/'
                self._create_directory(directory)
                filename = f'{directory}Frequency_Spectrum_Comparison_given_channels.pdf'
                filename = filename.replace(' ', '_')
                plt.savefig(filename, bbox_inches='tight', format='pdf')
                print(f'Saving to: {filename}')
                plt.close(fig)

                # Plot the spectra for predicted channels
                fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(self.page_width * 1.5, self.page_height * 0.75))
                axes = axes.flatten()

                handles_list = []
                labels_list = []

                for idx, title in enumerate(spectra_predicted.keys()):
                    if idx >= 7:
                        break  # Ensure we don't create more than 7 subplots

                    ax = axes[idx]
                    freq = np.fft.fftfreq(len(spectra_predicted[title][0]), d=1 / 2048.0)
                    line1, = ax.plot(freq, spectra_predicted[title][0], label='Original spectrum', color=spectrum_color,
                                     linewidth=0.5)
                    line2, = ax.plot(freq, spectra_predicted[title][1], label='Spectrum error', color=difference_color,
                                     linewidth=0.5)
                    ax.set_xlim(0, 600)  # Limiting the frequency range for better visualization
                    ax.set_title(f'{title}')
                    ax.grid(True)
                    ax.set_ylabel('Magnitude')
                    ax.set_xlabel('Frequency in Hz')

                    if idx == 0:
                        handles_list.append(line1)
                        handles_list.append(line2)
                        labels_list.append('Original spectrum')
                        labels_list.append('Spectrum error')

                legend_handles = [plt.Line2D([], [], color=handle.get_color(), linewidth=1.5) for handle in
                                  handles_list]

                legend_ax = axes[8]
                legend_ax.axis('off')
                legend_ax.legend(handles=legend_handles, labels=labels_list, loc='center')

                for i in range(idx + 1, 9):
                    axes[i].axis('off')

                fig.suptitle(f'Error - Predicted channels')
                fig.tight_layout()

                filename = f'{directory}Frequency_Spectrum_Comparison_predicted_channels.pdf'
                filename is filename.replace(' ', '_')
                plt.savefig(filename, bbox_inches='tight', format='pdf')
                print(f'Saving to: {filename}')
                plt.close(fig)

                #TODO: Separate plots for predicted and given
                # # Plot the spectra for the specific given channel (4, 2)
                # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(self.page_width * 1.5, self.page_height * 0.75))
                # axes = axes.flatten()
                #
                # handles_list = []
                # labels_list = []
                #
                # for idx, title in enumerate(spectra_given_positions.keys()):
                #     if idx >= 7:
                #         break  # Ensure we don't create more than 7 subplots
                #
                #     ax = axes[idx]
                #     freq = np.fft.fftfreq(len(spectra_given_positions[title][0]), d=1 / 2048.0)
                #     line1, = ax.plot(freq, spectra_given_positions[title][0], label='Original spectrum',
                #                      color=spectrum_color,
                #                      linewidth=0.5)
                #     line2, = ax.plot(freq, spectra_given_positions[title][1], label='Spectrum error',
                #                      color=difference_color,
                #                      linewidth=0.5)
                #     ax.set_xlim(0, 600)  # Limiting the frequency range for better visualization
                #     ax.set_title(f'{title}')
                #     ax.grid(True)
                #     ax.set_ylabel('Magnitude')
                #     ax.set_xlabel('Frequency in Hz')
                #     ax.set_ylim(-100000,1000000)  # Set global y-limits
                #
                #     if idx == 0:
                #         handles_list.append(line1)
                #         handles_list.append(line2)
                #         labels_list.append('Original spectrum')
                #         labels_list.append('Spectrum error')
                #
                # legend_handles = [plt.Line2D([], [], color=handle.get_color(), linewidth=1.5) for handle in
                #                   handles_list]
                #
                # legend_ax = axes[8]
                # legend_ax.axis('off')
                # legend_ax.legend(handles=legend_handles, labels=labels_list, loc='center')
                #
                # for i in range(idx + 1, 9):
                #     axes[i].axis('off')
                #
                # fig.suptitle(f'Error given channel')
                # fig.tight_layout()
                #
                # filename = f'{directory}Frequency_Spectrum_Comparison_given_channel_4_2.svg'
                # filename = filename.replace(' ', '_')
                # plt.savefig(filename, bbox_inches='tight', format='svg')
                # print(f'Saving to: {filename}')
                # plt.close(fig)
                #
                # # Plot the spectra for the specific predicted channel (3, 1)
                # fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(self.page_width * 1.5, self.page_height * 0.75))
                # axes = axes.flatten()
                #
                # handles_list = []
                # labels_list = []
                #
                # for idx, title in enumerate(spectra_predicted_positions.keys()):
                #     if idx >= 7:
                #         break  # Ensure we don't create more than 7 subplots
                #
                #     ax = axes[idx]
                #     freq = np.fft.fftfreq(len(spectra_predicted_positions[title][0]), d=1 / 2048.0)
                #     line1, = ax.plot(freq, spectra_predicted_positions[title][0], label='Original spectrum',
                #                      color=spectrum_color,
                #                      linewidth=0.5)
                #     line2, = ax.plot(freq, spectra_predicted_positions[title][1], label='Spectrum error',
                #                      color=difference_color,
                #                      linewidth=0.5)
                #     ax.set_xlim(0, 600)  # Limiting the frequency range for better visualization
                #     ax.set_title(f'{title}')
                #     ax.grid(True)
                #     ax.set_ylabel('Magnitude')
                #     ax.set_xlabel('Frequency in Hz')
                #     ax.set_ylim(-100000,1000000)  # Set global y-limits
                #
                #     if idx == 0:
                #         handles_list.append(line1)
                #         handles_list.append(line2)
                #         labels_list.append('Original spectrum')
                #         labels_list.append('Spectrum error')
                #
                # legend_handles = [plt.Line2D([], [], color=handle.get_color(), linewidth=1.5) for handle in
                #                   handles_list]
                #
                # legend_ax = axes[8]
                # legend_ax.axis('off')
                # legend_ax.legend(handles=legend_handles, labels=labels_list, loc='center')
                #
                # for i in range(idx + 1, 9):
                #     axes[i].axis('off')
                #
                # fig.suptitle(f'Error predicted channel')
                # fig.tight_layout()
                #
                # filename = f'{directory}Frequency_Spectrum_Comparison_predicted_channel_3_1.svg'
                # filename = filename.replace(' ', '_')
                # plt.savefig(filename, bbox_inches='tight', format='svg')
                # print(f'Saving to: {filename}')
                # plt.close(fig)

                # Create a single figure for both positions with subplots for all models
                fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(self.page_width * 1, self.page_height * 1))
                axes = axes.flatten()

                # Initialize lists for handles and labels for the legend
                handles_list = []
                labels_list = []

                # Plot the spectra for the specific given channel (4, 2)
                for idx, title in enumerate(spectra_given_positions.keys()):
                    ax = axes[2 * idx]
                    freq = np.fft.fftfreq(len(spectra_given_positions[title][0]), d=1 / 2048.0)
                    line1, = ax.plot(freq, spectra_given_positions[title][0], label='Original spectrum',
                                     color=spectrum_color,
                                     linewidth=0.5)
                    line2, = ax.plot(freq, spectra_given_positions[title][1], label='Spectrum error',
                                     color=difference_color,
                                     linewidth=0.5)
                    ax.set_xlim(0, 600)  # Limiting the frequency range for better visualization
                    ax.set_ylim(-100000, 1000000)  # Set global y-limits
                    ax.set_title(f'{title}')
                    ax.grid(True)
                    if idx == 0:
                        handles_list.append(line1)
                        handles_list.append(line2)
                        labels_list.append('Original spectrum')
                        labels_list.append('Spectrum error')
                    if idx == 6:  # Add x-axis label only to the bottom plots
                        ax.set_xlabel('Frequency in Hz')
                    else:
                        ax.set_xticklabels([])  # Remove x-axis tick labels for all but the bottom row
                    ax.set_ylabel('Magnitude')

                # Plot the spectra for the specific predicted channel (3, 1)
                for idx, title in enumerate(spectra_predicted_positions.keys()):
                    ax = axes[2 * idx + 1]
                    freq = np.fft.fftfreq(len(spectra_predicted_positions[title][0]), d=1 / 2048.0)
                    line1, = ax.plot(freq, spectra_predicted_positions[title][0], label='Original spectrum',
                                     color=spectrum_color,
                                     linewidth=0.5)
                    line2, = ax.plot(freq, spectra_predicted_positions[title][1], label='Spectrum error',
                                     color=difference_color,
                                     linewidth=0.5)
                    ax.set_xlim(0, 600)  # Limiting the frequency range for better visualization
                    ax.set_ylim(-100000, 1000000)  # Set global y-limits
                    ax.set_title(f'{title}')
                    ax.grid(True)
                    if idx == 6:  # Add x-axis label only to the bottom plots
                        ax.set_xlabel('Frequency in Hz')
                    else:
                        ax.set_xticklabels([])  # Remove x-axis tick labels for all but the bottom row
                    ax.set_yticklabels([])  # Remove y-axis tick labels for second column

                # Add a single legend in the middle below the two subplots
                legend_ax = axes[0]  # Using the last subplot for the legend
                legend = legend_ax.legend(handles=handles_list, labels=labels_list, loc='upper right')

                # Change the linewidth of the legend lines
                for line in legend.get_lines():
                    line.set_linewidth(1.5)

                fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend

                fig.text(0.29, 0.95, 'Spectrum - Given channel', ha='center', fontsize=10)
                fig.text(0.74, 0.95, 'Spectrum - Predicted channel', ha='center', fontsize=10)

                # Save the combined figure for given and predicted channels
                directory = f'plot_frequency_spectrum_difference/{path}/'
                self._create_directory(directory)
                filename = f'{directory}Frequency_Spectrum_Comparison_given_and_predicted_channels.pdf'
                filename = filename.replace(' ', '_')
                plt.savefig(filename, bbox_inches='tight', format='pdf')
                print(f'Saving to: {filename}')
                plt.close(fig)

        print("Finished plotting frequency spectrum differences.")


if __name__ == '__main__':
    file_path = '../results/'
    visualize = Visualizer(file_path)
    # visualize.plot_emg_channels()
    #visualize.plot_emg_channels_difference()
    #visualize.plot_pearson_coefficient_grid()
    visualize.create_difference_plots()
    # visualize.propagation_plot()
    # visualize.plot_windowed_correlations()
    #visualize.plot_specific_positions()
    #visualize.create_box_plot()
    #visualize.plot_frequency_spectrum_difference()
    # visualize.plot_normalized_correlations()
    #visualize.process_matlab_rms()