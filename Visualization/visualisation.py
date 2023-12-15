import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt


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

        figure = 1

        for title, emg_data in self.emg_data_dict.items():
            # Plot Training losses in a separate figure
            plt.figure(figure)
            figure += 1

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

    def plot_heat_maps(self):
        grids = (
            self.emg_data_dict['Origin HR'],
            self.emg_data_dict['Interpolation LR to SR'],
            self.emg_data_dict['Model 1 LR to SR']
        )

        keys_list = ('Origin HR', 'Interpolation LR to SR', 'Model 1 LR to SR')

        time_points = [200, 300, 400]

        fig, axs = plt.subplots(len(grids), len(time_points))

        for g in range(len(grids)):
            emg_data = grids[g]
            emg_data = emg_data.transpose()
            emg_data_reshaped = emg_data.reshape(8, 8, 512)
            emg_data_transposed = np.transpose(emg_data_reshaped, axes=(1, 0, 2))
            emg_data_grid = np.flipud(emg_data_transposed)

            for i, time_point in enumerate(time_points):
                emg_heatmap = emg_data_grid[:, :, time_point]
                im = axs[g][i].imshow(emg_heatmap, cmap='hot')

                # Set ticks and tick labels to start from 1 but position them at 0 to match the data
                axs[g][i].set_xticks(range(8))
                axs[g][i].set_yticks(range(8))
                axs[g][i].set_xticklabels(range(1, 9))
                axs[g][i].set_yticklabels(range(1, 9))

                axs[g][i].invert_yaxis()  # Invert the y-axis direction
                fig.colorbar(im, ax=axs[g][i])

        plt.tight_layout()  # Adjust layouts to avoid overlapping
        plt.savefig(f'plot_heatmaps/Heatmaps.png')

    def plot_correlations(self):
        grids = (self.emg_data_dict['Origin HR'], self.emg_data_dict['Interpolation LR to SR'],
                 self.emg_data_dict['Model 1 LR to SR'])
        # Accessing keys by index
        keys_list = ('Origin HR', 'Interpolation LR to SR', 'Model 1 LR to SR')

        # Create a list of line styles, colors
        colors = ['blue', 'green', 'red', 'black']  # Add more colors if required
        line_styles = ['-', '--']  # You can extend this list as needed

        for g in range(1, len(grids)):
            grid_1 = grids[0]
            grid_2 = grids[g]

            # Calculate cross-correlation for specific pairs of electrodes
            correlations = []
            for i in range(64):  # Iterate through electrodes
                correlation = np.correlate(grid_1[i], grid_2[i], mode='same')
                correlations.append(correlation)

            # Find the maximum and minimum values of correlations across all electrodes
            max_corr = max([max(corr) for corr in correlations])
            min_corr = min([min(corr) for corr in correlations])

            # Plot the correlations
            plt.figure()
            color = 0
            style = 0
            number = 1
            for i, corr in enumerate(correlations):
                plt.plot(corr, label=f'Ch {i + 1}', linestyle=line_styles[style % len(line_styles)],
                         color=colors[color % len(colors)])  #
                plt.xlabel('Shift')
                plt.ylabel('Correlation')
                plt.legend(loc='best', bbox_to_anchor=(1, 1))
                plt.title(f'Cross-correlation of {keys_list[g]} with Origin HR')

                # Set the y-axis limits to the maximum and minimum values across all electrodes
                plt.ylim(min_corr, max_corr)

                color += 1
                if color == 4:
                    color = 0
                    style += 1

                if ((i + 1) % 8) == 0:
                    plt.savefig(f'plot_correlations/{keys_list[g]} with Origin HR - {number}.png')
                    number += 1
                    plt.close()
                    plt.figure()
                    color = 0
                    style = 0


if __name__ == '__main__':
    file_path = '../testset/Dominik/offline/20230720-091659'
    visualize = Visualizer(file_path)

    visualize.plot_emg_channels()
    visualize.plot_correlations()
    visualize.plot_heat_maps()
