import json
import os
import pickle
import pandas as pandas
import matplotlib

matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.io import loadmat


def plot_emg_data(emg_data):
    """

    Parameters
    ----------
    emg_data: ndarray
              emg signal as an array

    Returns
    -------
    None
    """

    print("emg_data shape", emg_data.shape)
    time_axis = np.arange(512)  # Assuming the time axis runs along axis 2

    plt.figure(figsize=(10, 6))

    offset = 0
    for i in range(emg_data.shape[0]):
        for j in range(emg_data.shape[1]):
            signal = emg_data[i, j]
            print("shape", signal.shape)

            signal = np.squeeze(signal)  # squeeze to go from size (1,91441) to (91441,)
            signal = signal[:512]
            plt.plot(time_axis, np.squeeze(signal) + offset, label=f'Channel ({i + 1}, {j + 1})')
            offset += 500

    plt.xlabel('Time')
    plt.ylabel('Channel')
    plt.title('EMG Signals')
    plt.legend(loc='right')
    plt.show()


def open_emg_data_Dominik(file_path):
    """
    Opens the emg signal of from a pickle file in a given directory
    Parameters
    ----------
    file_path: str
                 directory of the pickle file
    Returns
    -------
    emg_data: ndarray
              EMG signal of 128 electrodes
    """

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Dominiks data:
    emg_data = data['_input_signal']

    return emg_data


def dataset_creator_dominik(data_paths):
    """
    Creates sets of low-resolution and high-resolution data as well as a csv file with their directories.
    Parameters
    ----------
    data_paths: list of str
                A list of multiple directories, each containing a pickle file

    Returns
    -------
    None
    """

    # List of LR and HR filenames (replace this with your filenames)
    lr_filenames = []
    hr_filenames = []

    min_value_dataset = np.inf
    max_value_dataset = -np.inf

    combined_data = []

    for i in range(len(data_paths)):

        # extract data path and pickle file name from list of paths
        data_path = data_paths[i]

        # create folder for the results
        base_directory = data_path.replace("data/", "").rsplit("/", 1)[0]
        dataset_path = 'dataset' + '/' + base_directory + '/'

        # Create the folder if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)

        # get the emg data of that pickle file
        emg_data_complete = open_emg_data_Dominik(data_path)

        plt.figure(i)

        # Creating a histogram using numpy.histogram()
        counts, bins = np.histogram(emg_data_complete, bins=20)  # Adjust bins as needed

        # Plotting the histogram
        plt.stairs(counts, bins)
        plt.xlabel('Data')
        plt.ylabel('Density')
        plt.title('Histogram Data')
        plt.savefig(dataset_path + f'Histogram-path-{i}-20bins.png')

        for grid in range(2):
            # only use first 64 channels, since data is from two grids
            emg_data = emg_data_complete[grid * 64:(grid * 64) + 64, :]

            # 2D to 3D:
            # reshape back to grid shape
            emg_data_reshaped = emg_data.reshape(8, 8, emg_data.shape[1])
            # Transpose and flip to get to correct grid numbering
            emg_data_transposed = np.transpose(emg_data_reshaped, axes=(1, 0, 2))
            emg_data_grid = np.flipud(emg_data_transposed)

            windows = emg_data.shape[1] - 512

            # window the signal to 512 samples
            for window in range(0, windows, 50):
                emg_windowed = emg_data_grid[:, :, window:(window + 512)]

                if emg_windowed.min() < min_value_dataset:
                    min_value_dataset = emg_windowed.min()

                if emg_windowed.max() > max_value_dataset:
                    max_value_dataset = emg_windowed.max()

                for col in range(2):
                    for row in range(2):
                        emg_windowed_hr = emg_windowed[row:row + 7, col:col + 7, :]
                        emg_windowed_lr = emg_windowed[row::2, col::2, :]

                        # 3D to 2D array
                        emg_windowed_lr_2D = np.flipud(emg_windowed_lr).transpose(1, 0, 2).reshape(16, 512)
                        emg_windowed_hr_2D = np.flipud(emg_windowed_hr).transpose(1, 0, 2).reshape(49, 512)

                        combined_data.append(emg_windowed_hr_2D)

                        # transpose for samples x channels shape
                        emg_windowed_lr_2D = emg_windowed_lr_2D.transpose()
                        emg_windowed_hr_2D = emg_windowed_hr_2D.transpose()

                        # save high resolution data and add file name to list
                        np.save(dataset_path + f'grid-{grid}-window{window}-pos-{col}-{row}-hr.npy', emg_windowed_hr_2D)
                        hr_filenames.append('../' + dataset_path + f'grid-{grid}-window{window}-pos-{col}-{row}-hr.npy')

                        # save low resolution data and add file name to list
                        np.save(dataset_path + f'grid-{grid}-window{window}-pos-{col}-{row}-lr.npy', emg_windowed_lr_2D)
                        lr_filenames.append('../' + dataset_path + f'grid-{grid}-window{window}-pos-{col}-{row}-lr.npy')

    # Create a dictionary for the hyperparameters
    min_max_dic = {
        "min": min_value_dataset,
        "max": max_value_dataset,
    }

    mean = np.mean(combined_data)
    std = np.std(combined_data)

    # Create a dictionary for the hyperparameters
    mean_std_dic = {
        "mean": mean,
        "std": std,
    }

    with open('dataset' + '/min_max.json', 'w') as file:
        json.dump(min_max_dic, file)

    with open('dataset' + '/mean_std.json', 'w') as file:
        json.dump(mean_std_dic, file)

    # Combine LR and HR filenames row-wise
    data = zip(lr_filenames, hr_filenames)

    # File path for the CSV file
    csv_file = 'data.csv'

    # Open the CSV file in append mode and write the data
    with open('dataset' + '/' + csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Add a header if the file is empty
        if file.tell() == 0:
            writer.writerow(['lr_filename;hr_filename'])

        # Add each LR/HR pair as a row in the CSV file
        for lr_filename, hr_filename in data:
            writer.writerow([lr_filename + ';' + hr_filename])


def dataset_creator_alessandro(data_paths):
    """

    Parameters
    ----------
    data_paths: list of str

    Returns
    -------
    None
    """

    # List of LR and HR filenames (replace this with your filenames)
    lr_filenames = []
    hr_filenames = []

    for i in range(len(data_paths)):
        # extract data path and pickle file name from list of paths
        data_path = data_paths[i]

        # Load MATLAB file
        mat_contents = loadmat(data_path)

        # Get the data from the MATLAB file
        emg_data = mat_contents['SIG']

        # create folder for the results in dataset/Alessandro_sEMG/AINN_DECOMP_PROX_35-50-70_FINISHED_TO/PRE
        base_directory = data_path.replace("data/", "").rsplit("/", 1)[0]
        dataset_path = 'dataset' + '/' + base_directory + '/'
        print(dataset_path)

        # Create the folder if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)

        emg_lr = emg_data[::2, ::2]

        emg_length = emg_lr[0][0][0].shape[0]
        emg_hr_reshaped = np.zeros((emg_data.shape[0], emg_data.shape[1], emg_length))
        emg_lr_reshaped = np.zeros((emg_lr.shape[0], emg_lr.shape[1], emg_length))
        print("shape emg_lr_reshaped:", np.shape(emg_lr_reshaped))

        # Reshape HR data into 3D
        for row in range(emg_data.shape[0]):
            for col in range(emg_data.shape[1]):
                if (row == (emg_data.shape[0] - 1) and col == 0):
                    continue
                else:
                    sub_array = emg_data[row][col]
                    emg_hr_reshaped[row, col] = sub_array

        # Reshape LR data into 3D
        for row in range(emg_lr.shape[0]):
            for col in range(emg_lr.shape[1]):
                if (row == (emg_lr.shape[0] - 1) and col == 0):
                    continue
                else:
                    sub_array = emg_lr[row][col]
                    emg_lr_reshaped[row, col] = sub_array


        # Reshape arrays to channel x samples, counting channels from top to bottom
        emg_lr_reshaped_final = np.flipud(emg_lr_reshaped).transpose(1, 0, 2).reshape(-1, emg_lr_reshaped.shape[-1])
        emg_hr_reshaped_final = np.flipud(emg_hr_reshaped).transpose(1, 0, 2).reshape(-1, emg_hr_reshaped.shape[-1])

        # Get mean and std for standardization
        mean = np.mean(emg_hr_reshaped_final)
        std = np.std(emg_hr_reshaped_final)

        # Create a dictionary for the hyperparameters
        mean_std_dict = {
            "mean": mean,
            "std": std,
        }

        with open('dataset' + '/mean_std.json', 'w') as file:
            json.dump(mean_std_dict, file)

        windows = emg_length - 512

        # window the signal to 512 samples
        for window in range(0, windows):
            print("window", window)
            emg_lr_windowed = emg_lr_reshaped_final[:, window:(window + 512)]
            emg_hr_windowed = emg_hr_reshaped_final[:, window:(window + 512)]

            # Delete first channel since it's empty
            emg_lr_final = emg_lr_windowed[1:]
            emg_hr_final = emg_hr_windowed[1:]

            # transpose for samples x channels shape
            emg_lr_final = emg_lr_final.transpose()
            emg_hr_final = emg_hr_final.transpose()

            # save high resolution data and add file name to list
            np.save(dataset_path + f'window{window}-hr.npy', emg_hr_final)
            hr_filenames.append('../' + dataset_path + f'window{window}-hr.npy')

            # save low resolution data and add file name to list
            np.save(dataset_path + f'window{window}-lr.npy', emg_lr_final)
            lr_filenames.append('../' + dataset_path + f'window{window}-lr.npy')

    # Combine LR and HR filenames row-wise
    data = zip(lr_filenames, hr_filenames)
    # File path for the CSV file
    csv_file = 'data.csv'

    # Open the CSV file in append mode and write the data
    with open('dataset' + '/' + csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Add a header if the file is empty
        if file.tell() == 0:
            writer.writerow(['lr_filename;hr_filename'])

        # Add each LR/HR pair as a row in the CSV file
        for lr_filename, hr_filename in data:
            writer.writerow([lr_filename + ';' + hr_filename])


if __name__ == '__main__':
    # data_paths = ['data/Dominik/offline/20230720-085154/Ambra_20230720-085154_Follow Trajectory.pkl',
    #               'data/Dominik/offline/20230720-090835/Ambra_20230720-090835_Follow Trajectory.pkl',
    #               'data/Dominik/offline/20230720-091107/Ambra_20230720-091107_Follow Trajectory.pkl']
    # # data_paths = ['data/Dominik/offline/20230720-091659/Ambra_20230720-091659_Follow Trajectory.pkl']
    #
    # movement_name = 'thumb_slow'
    #
    # dataset_creator_dominik(data_paths)

    data_paths = [
        'data/Alessandro_sEMG/AINN_DECOMP_PROX_35-50-70_FINISHED_TO/PRE/400InAnfd170605141107 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length27_runs250_FINISHED_ANDREA.mat']

    dataset_creator_alessandro(data_paths)
