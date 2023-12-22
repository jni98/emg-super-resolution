import json
import os
import pickle
import pandas as pandas
import matplotlib

matplotlib.use('TkAgg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt
import csv
import numpy as np


def open_all_files_for_one_patient_and_movement(folder_path, movement_name, pickle_file_name):
    """
    opens all files (EMG, MU, ref) for one patient and one movement and returns the data as a dictionary
    :param folder_path: path to the folder that contains data pkl and the 12 csv files for kinematics
    :param movement_name: which movement you want to open all the files for
    :param pickle_file_name:
    :return: dictionarys for emg_data, MU_data, ref_data
    """

    movement_list = ["thumb_slow", "thumg_fast", "indec_slow", "index_fast", "middle_slow", "middle_fast", "ring_slow",
                     "ring_fast", "pinky_slow", "pinky_fast", "fist", "2pinch", "3pinch"]
    movement_number = movement_list.index(movement_name) + 1
    file_path = os.path.join(folder_path, pickle_file_name)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    emg_data = data[movement_name]["SIG"]
    MU_data = data[movement_name]["MUPulses"]
    ref_file = os.path.join(folder_path, str(movement_number) + "_transformed.csv")
    ref_data = pandas.read_csv(ref_file)

    return emg_data, MU_data, ref_data


def open_emg_data_Michael(folder_path, movement_name, pickle_file_name):
    """
    Opens the emg signal of a defined movement
    Parameters
    ----------
    folder_path: str
                 directory of the pickle file
    movement_name : str
                 the movement you want to open the emg for
    pickle_file_name : str
                 name of the pickle file to restore emg signal

    Returns
    -------
    emg_data: ndarray
              EMG signal of electrode Grids 1-5
    """

    file_path = os.path.join(folder_path, pickle_file_name)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Michaels Data
    emg_data = data[movement_name]["SIG"]

    return emg_data


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


def dataset_creator_michi(emg_data, dataset_path):
    """
    Creates the dataset including a csv file from the emg_data. Used for the data provided by Michi
    Parameters
    ----------
    emg_data: ndarray
              emg signal as a 3D array
    dataset_path: str
                  path where the results should be saved

    Returns
    -------
    None
    """

    # TODO: adapt to other method where multiple paths are given as input

    # Create the folder if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)

    # List of LR and HR filenames (replace this with your filenames)
    lr_filenames = []
    hr_filenames = []

    # go through electrode blockgs G1, G2, G3
    for block in range(3):
        emg_block = emg_data[0:8, 1 + block * 8:9 + block * 8]

        # window the signal to 512 samples
        for window in range(50, 100):
            emg_windowed = np.zeros((emg_block.shape[0], emg_block.shape[1], 512))

            for row in range(8):
                for column in range(8):
                    # squeeze because of shape (1,512)
                    squeezed = np.squeeze(emg_block[row, column])
                    emg_windowed[row, column] = squeezed[window * 512:(window + 1) * 512]

            # save high resolution data and add file name to list
            np.save(dataset_path + f'G{block + 1}-window{window}-hr.npy', emg_windowed)
            hr_filenames.append(dataset_path + f'G{block + 1}-window{window}-hr.npy')

            # TODO: bei lr version immer unterschiedliche Elektroden nehmen

            # Create low resolution of windowed data, by using only every second row and column
            emg_windowed_lr = emg_windowed[::2, ::2, :]

            # save low resolution data and add file name to list
            np.save(dataset_path + f'G{block + 1}-window{window}-lr.npy', emg_windowed_lr)
            lr_filenames.append(dataset_path + f'G{block + 1}-window{window}-lr.npy')

    # Combine LR and HR filenames row-wise
    data = zip(lr_filenames, hr_filenames)

    # File path for the CSV file
    csv_file = 'data.csv'

    # Open the CSV file in append mode and write the data
    with open(dataset_path + csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Add a header if the file is empty
        if file.tell() == 0:
            writer.writerow(['lr_filename', 'hr_filename'])

        # Add each LR/HR pair as a row in the CSV file
        for lr_filename, hr_filename in data:
            writer.writerow([lr_filename, hr_filename])


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

        # only use first 64 channels, since data is from two grids
        emg_data = emg_data_complete[:64, :]

        # 2D to 3D:
        # reshape back to grid shape
        emg_data_reshaped = emg_data.reshape(8, 8, emg_data.shape[1])
        # Transpose and flip to get to correct grid numbering
        emg_data_transposed = np.transpose(emg_data_reshaped, axes=(1, 0, 2))
        emg_data_grid = np.flipud(emg_data_transposed)

        windows = emg_data.shape[1] - 512

        # window the signal to 512 samples
        for window in range(0, windows, 200):
            emg_windowed = emg_data_grid[:, :, window:(window + 512)]

            if emg_windowed.min() < min_value_dataset:
                min_value_dataset = emg_windowed.min()

            if emg_windowed.max() > max_value_dataset:
                max_value_dataset = emg_windowed.max()

            emg_windowed_lr = emg_windowed[::2, ::2, :]

            # 3D to 2D array
            emg_windowed_lr_2D = np.flipud(emg_windowed_lr).transpose(1, 0, 2).reshape(16, 512)
            emg_windowed_2D = np.flipud(emg_windowed).transpose(1, 0, 2).reshape(64, 512)

            # transpose for samples x channels shape
            emg_windowed_lr_2D = emg_windowed_lr_2D.transpose()
            emg_windowed_2D = emg_windowed_2D.transpose()

            # save high resolution data and add file name to list
            np.save(dataset_path + f'window{window}-hr.npy', emg_windowed_2D)
            hr_filenames.append('../' + dataset_path + f'window{window}-hr.npy')

            # save low resolution data and add file name to list
            np.save(dataset_path + f'window{window}-lr.npy', emg_windowed_lr_2D)
            lr_filenames.append('../' + dataset_path + f'window{window}-lr.npy')

    # Create a dictionary for the hyperparameters
    min_max_dic = {
        "min": min_value_dataset,
        "max": max_value_dataset,
    }

    with open('dataset' + '/min_max.json', 'w') as file:
        json.dump(min_max_dic, file)


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
    data_paths = ['data/Dominik/offline/20230720-085154/Ambra_20230720-085154_Follow Trajectory.pkl',
                  'data/Dominik/offline/20230720-090835/Ambra_20230720-090835_Follow Trajectory.pkl',
                  'data/Dominik/offline/20230720-091107/Ambra_20230720-091107_Follow Trajectory.pkl']
    # data_paths = ['data/Dominik/offline/20230720-091659/Ambra_20230720-091659_Follow Trajectory.pkl']

    movement_name = 'thumb_slow'

    dataset_creator_dominik(data_paths)
