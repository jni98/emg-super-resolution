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
    opens all files (EMG, MU, ref) for one patient and one movement and returns the data as a dictionary
    :param folder_path: path to the folder that contains data pkl and the 12 csv files for kinematics
    :param movement_name: which movement you want to open all the files for
    :param pickle_file_name:
    :return: dictionarys for emg_data
    """

    file_path = os.path.join(folder_path, pickle_file_name)
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Michaels Data
    emg_data = data[movement_name]["SIG"]

    return emg_data


def plot_emg_data(emg_data):
    """
        plots all channels by a offset of 500
        :param emg_data: extracted emg_data
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


def flatten_array_test(array):
    """

    Parameters
    ----------
    array: given array

    Returns: array that is flattend in column-major bottom up
    -------

    """

    flattened_array = np.flip(array, axis=0).flatten('F')  # Reverse the flattened array

    return flattened_array


def dataset_creator_michi(emg_data, dataset_path):
    """
    Creates the dataset including a csv file from the emg_data. Used for the data provided by Michi
    Parameters
    :param emg_data
    :param data_path: path to the folder that contains data pkl and the 12 csv files for kinematics
    :param dataset_path
    """

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
    opens all files (EMG, MU, ref) for one patient and one movement and returns the data as a dictionary
    :param folder_path: path to the folder that contains data pkl and the 12 csv files for kinematics
    :param movement_name: which movement you want to open all the files for
    :param pickle_file_name:
    :return: dictionarys for emg_data
    """

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # Dominiks data:
    emg_data = data['_input_signal']

    return emg_data


def dataset_creator_dominik(data_paths):
    """
    Creates the dataset including a csv file from the emg_data. Used for the data provided by Michi
    Parameters
    :param emg_data
    :param data_path: path to the folder that contains data pkl and the 12 csv files for kinematics
    :param dataset_path
    """
    # List of LR and HR filenames (replace this with your filenames)
    lr_filenames = []
    hr_filenames = []

    for i in range(len(data_paths)):

        # extract data path and pickle file name from list of paths
        data_path = data_paths[i]
        pickle_file_name = os.path.basename(data_path)

        # create folder for the results
        base_directory = data_path.replace("../data/", "").rsplit("/", 1)[0]
        dataset_path = 'dataset' + '/' + base_directory + '/'

        # get the emg data of that pickle file
        emg_data = open_emg_data_Dominik(data_path)

        # Create the folder if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)

        lr_mode = 0

        # window the signal to 512 samples
        for window in range(1, 75):
            emg_windowed = emg_data[:, window * 512:(window + 1) * 512]

            # To keep different electrodes in lr version
            if lr_mode == 4:
                lr_mode = 0

            # Create low resolution of windowed data, by using only every second row and column
            emg_windowed_lr = emg_windowed[lr_mode::4, :]
            lr_mode += 1

            emg_windowed = emg_windowed.transpose()
            emg_windowed_lr = emg_windowed_lr.transpose()

            # save high resolution data and add file name to list
            np.save(dataset_path + f'window{window}-hr.npy', emg_windowed)
            hr_filenames.append(dataset_path + f'window{window}-hr.npy')

            # save low resolution data and add file name to list
            np.save(dataset_path + f'window{window}-lr.npy', emg_windowed_lr)
            lr_filenames.append(dataset_path + f'window{window}-lr.npy')

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
    data_paths = ['../data/Dominik/offline/20230720-085154/Ambra_20230720-085154_Follow Trajectory.pkl',
                  '../data/Dominik/offline/20230720-090835/Ambra_20230720-090835_Follow Trajectory.pkl']

    movement_name = 'thumb_slow'

    dataset_creator_dominik(data_paths)
