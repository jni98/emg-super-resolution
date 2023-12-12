import numpy as np
import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot
from matplotlib import pyplot as plt


class Visualizer:
    def __init__(self, file_path):
        self.file_path = file_path

    def plot_emg_channels(self, emg_data, type):

        print("emg_data shape", emg_data.shape)
        time_axis = np.arange(512)  # Assuming the time axis runs along axis 2

        emg_data = emg_data.transpose()
        print("emg data shape", emg_data.shape)

        # Plot Training losses in a separate figure
        fig_emg_channels, ax = plt.subplots()

        offset = 0
        for i in range(emg_data.shape[0]):
            signal = emg_data[i]

            ax.plot(time_axis, signal.squeeze() + offset, label=f'Channel {i + 1}')
            offset += 500

        ax.set_xlabel('Time')
        ax.set_ylabel('Channel')
        ax.set_title(type)
        ax.legend(loc='best')
        fig_emg_channels.savefig(f'plot_channels/{type}.png')
        plt.figure(fig_emg_channels)
        plt.show()

    def plot_heat_maps(self, emg_data):
        pass

    def plot_correlations(self, emg_data):
        pass

if __name__ == '__main__':
    file_path = '../testset/Dominik/offline/20230720-091659'
    visualize = Visualizer(file_path)

    inter_lr_sr = np.load('../testset/Dominik/offline/20230720-091659/interpolated/bspline/window10-lr-sr.npy')
    inter_hr_sr = np.load('../testset/Dominik/offline/20230720-091659/interpolated/bspline/window10-hr-sr.npy')
    model1_lr_sr= np.load('../testset/Dominik/offline/20230720-091659/model1/window10-lr-sr.npy')
    model1_hr_sr= np.load('../testset/Dominik/offline/20230720-091659/model1/window10-hr-sr.npy')
    origin_hr = np.load('../testset/Dominik/offline/20230720-091659/window10-hr.npy')
    origin_lr = np.load('../testset/Dominik/offline/20230720-091659/window10-lr.npy')

    visualize.plot_emg_channels(inter_lr_sr, 'Interpolation LR to SR')
    visualize.plot_emg_channels(inter_hr_sr, 'Interpolation HR to SR')

    visualize.plot_emg_channels(model1_lr_sr, 'Model 1 LR to SR')
    visualize.plot_emg_channels(model1_hr_sr, 'Model 1 HR to SR')

    visualize.plot_emg_channels(origin_lr, 'Origin LR')
    visualize.plot_emg_channels(origin_hr, 'Origin HR')