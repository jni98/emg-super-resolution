import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot
from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self, file_path):
        self.file_path = file_path

        # Load your data here - replace these with your data loading logic
        inter_lr_sr = np.load(self.file_path + '/interpolated/bspline/window10-lr-sr.npy').transpose()
        inter_hr_sr = np.load(self.file_path + '/interpolated/bspline/window10-hr-sr.npy').transpose()
        model1_lr_sr = np.load(self.file_path + '/model1/window10-lr-sr.npy').transpose()
        model1_hr_sr = np.load(self.file_path + '/model1/window10-hr-sr.npy').transpose()
        origin_hr = np.load(self.file_path + '/window10-hr.npy').transpose()

        # Create a dictionary with titles as keys and EMG data as values
        self.emg_data_dict = {
            'Interpolation LR to SR': inter_lr_sr,
            'Interpolation HR to SR': inter_hr_sr,
            'Model 1 LR to SR': model1_lr_sr,
            'Model 1 HR to SR': model1_hr_sr,
            'Origin HR': origin_hr
        }

    def plot_emg_channels(self):
        time_axis = np.arange(512)  # Assuming the time axis runs along axis 2

        i = 1

        for title, emg_data in self.emg_data_dict.items():
            # Plot Training losses in a separate figure
            plt.figure(i)
            i += 1

            offset = 0
            # Use only few channels here
            for j in range(16):
                signal = emg_data[j]
                plt.plot(time_axis, signal.squeeze() + offset, label=f'Channel {j + 1}')
                offset += 1500

            plt.xlabel('Time')
            plt.ylabel('Voltage')
            plt.title(title)
            plt.legend(loc='best')
            plt.savefig(f'plot_channels/{title}.png')
        plt.show()

    def plot_heat_maps(self, emg_data):
        pass

    def plot_correlations(self):
        grids = (self.emg_data_dict['Origin HR'],  self.emg_data_dict['Interpolation LR to SR'], self.emg_data_dict['Model 1 LR to SR'])
        # Accessing keys by index
        keys_list = list(self.emg_data_dict.keys())

        # Create a list of line styles, colors, and markers
        colors = ['blue', 'green', 'red', 'black']  # Add more colors if required
        line_styles = ['-', '--']  # You can extend this list as needed
        markers = [',', 'x']  # Include additional markers for variety

        for g in range(1,len(grids)):
            grid_1 = grids[0]
            grid_2 = grids[g]

            # Calculate cross-correlation for specific pairs of electrodes
            correlations = []
            for i in range(64):  # Iterate through electrodes
                correlation = np.correlate(grid_1[i], grid_2[i], mode='same')
                correlations.append(correlation)

            # Plot the correlations
            fig, axes = plt.subplots(nrows=2, ncols=4, sharey=True)
            color = 0
            style = 0
            marker = 0
            for i, corr in enumerate(correlations):
                if i == 0:
                    ax = axes[0][0]
                    color = 0
                    style = 0
                    marker = 0
                elif i ==8:
                    ax = axes[0][1]
                    color = 0
                    style = 0
                    marker = 0
                elif i == 16:
                    ax = axes[0][2]
                    color = 0
                    style = 0
                    marker = 0
                elif i == 24:
                    ax = axes[0][3]
                    color = 0
                    style = 0
                    marker = 0
                elif i == 32:
                    ax = axes[1][0]
                    color = 0
                    style = 0
                    marker = 0
                elif i ==40:
                    ax = axes[1][1]
                    color = 0
                    style = 0
                    marker = 0
                elif i == 48:
                    ax = axes[1][2]
                    color = 0
                    style = 0
                    marker = 0
                elif i == 56:
                    ax = axes[1][3]
                    color = 0
                    style = 0
                    marker = 0


                ax.plot(corr, label=f'Ch {i + 1}', linestyle=line_styles[style % len(line_styles)],
                             color=colors[color % len(colors)], marker=markers[marker % len(markers)]) #
                color +=1
                if color == 4:
                    color = 0
                    style += 1

                if style == 3:
                    color = 0
                    style = 0
                    marker += 1

                ax.set_xlabel('Shift')
                ax.set_ylabel('Correlation')
                ax.legend(loc='best', bbox_to_anchor=(1, 1))
            fig.suptitle(f'Cross-correlation of {keys_list[g]} with Origin HR')
            fig.savefig(f'plot_correlations/{keys_list[g]}_with_Origin_HR.png')
        plt.show()


if __name__ == '__main__':
    file_path = '../testset/Dominik/offline/20230720-091659'
    visualize = Visualizer(file_path)

    visualize.plot_emg_channels()
    visualize.plot_correlations()
    input("Press Enter to close all plots...")