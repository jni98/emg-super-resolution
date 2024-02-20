"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains functions to open and plot EMG data and to create a dataset from
                    the EMG data including a csv file with their directories. The dataset_creator_dominik function
                    creates sets from a pickle file, the dataset_creator_alessandro function creates
                    the same for a matlab EMG data file.

You can run this code by defining the data paths of the raw data and calling the dataset_creator_dominik or
dataset_creator_alessandro function with the data paths as an argument.
"""

import json
import os
import pickle
import tarfile

import matplotlib

matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.io import loadmat


def plot_emg_data(emg_data: np.ndarray) -> None:
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
    plt.savefig('emg_signals.png')
    plt.close()


def open_emg_data_Dominik(file_path: str) -> np.ndarray:
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


def dataset_creator_dominik(data_paths: list[str]) -> None:
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
        plt.close(i)

        for grid in range(1):
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
            for window in range(0, 3000, 50):
                emg_windowed = emg_data_grid[:, :, window:(window + 512)]

                if emg_windowed.min() < min_value_dataset:
                    min_value_dataset = emg_windowed.min()

                if emg_windowed.max() > max_value_dataset:
                    max_value_dataset = emg_windowed.max()

                # range(2): uses full grid for generation of training data
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


def dataset_creator_alessandro_13x5(train_paths: list[str], test_paths: list[str]) -> None:
    """

    Parameters
    ----------
    train_paths: list of str
                A list of multiple directories, each containing a matlab file
    test_paths: list of str
                A list of multiple directories, each containing a matlab file

    Returns
    -------
    None
    """

    ######################## Training ########################

    EMGGainFactor = 5 / (2 ** 16) / 150 * 1000

    # List of LR and HR filenames (replace this with your filenames)
    lr_filenames_training = []
    hr_filenames_training = []

    for i in range(len(train_paths)):
        # extract data path
        data_path = train_paths[i]

        # Load MATLAB file
        mat_contents = loadmat(data_path)

        # Get the data from the MATLAB file
        emg_data = mat_contents['SIG']

        # convert from mV to µV
        emg_data = emg_data / EMGGainFactor

        # create folder for the results in dataset/Alessandro_sEMG/AINN_DECOMP_PROX_35-50-70_FINISHED_TO/PRE
        dataset_path = 'dataset' + '/train/'
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

        plt.figure()
        plt.plot(emg_hr_reshaped[0, 0])
        plt.savefig('emg_hr_reshaped.png')
        plt.close()

        # Reshape arrays to channel x samples, counting channels from top to bottom
        # emg_lr_reshaped_final = np.flipud(emg_lr_reshaped).transpose(1, 0, 2).reshape(-1, emg_lr_reshaped.shape[-1])
        # emg_hr_reshaped_final = np.flipud(emg_hr_reshaped).transpose(1, 0, 2).reshape(-1, emg_hr_reshaped.shape[-1])

        if i == 0:
            combined_data = emg_hr_reshaped.flatten()
        else:
            combined_data = np.concatenate((combined_data, emg_hr_reshaped.flatten()))

        windows = emg_length - 2048

        # window the signal to 2048 samples
        # 51 samples equals 25ms which is approximately 1 action potential --> 2AP overlapping
        for window in range(0, windows, (2048 - 102)):
            print("window", window)
            # The length of one sample is 2048 which is equal to 1 second
            emg_lr_windowed = emg_lr_reshaped[:, :, window:(window + 2048)]
            emg_hr_windowed = emg_hr_reshaped[:, :, window:(window + 2048)]

            # transpose for samples x channels shape
            # emg_lr_final = emg_lr_windowed.transpose()
            # emg_hr_final = emg_hr_windowed.transpose()

            # save high resolution data and add file name to list
            hr_file_location = dataset_path + f'path{i}-window{window}-hr.npy'
            np.save(hr_file_location, emg_hr_windowed)
            hr_filename = f'path{i}-window{window}-hr.npy'
            hr_filenames_training.append(hr_filename)

            # save low resolution data and add file name to list
            lr_file_location = dataset_path + f'path{i}-window{window}-lr.npy'
            np.save(lr_file_location, emg_lr_windowed)
            lr_filename = f'path{i}-window{window}-lr.npy'
            lr_filenames_training.append(lr_filename)

    # Get mean and std for standardization
    mean = np.mean(combined_data)
    std = np.std(combined_data)

    # Create a dictionary for the hyperparameters
    mean_std_dict = {
        "mean": mean,
        "std": std,
    }

    with open('dataset/train' + '/mean_std.json', 'w') as file:
        json.dump(mean_std_dict, file)

    # Combine LR and HR filenames row-wise
    data = zip(lr_filenames_training, hr_filenames_training)
    # File path for the CSV file
    csv_file = 'data.csv'

    # Open the CSV file in append mode and write the data
    with open('dataset/train' + '/' + csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Add a header if the file is empty
        if file.tell() == 0:
            writer.writerow(['lr_filename;hr_filename'])

        # Add each LR/HR pair as a row in the CSV file
        for lr_filename, hr_filename in data:
            writer.writerow([lr_filename + ';' + hr_filename])

    ######################## Test ########################

    # List of LR and HR filenames (replace this with your filenames)
    lr_filenames_test = []
    hr_filenames_test = []

    for i in range(len(test_paths)):
        # extract data path
        data_path = test_paths[i]

        # Load MATLAB file
        mat_contents = loadmat(data_path)

        # Get the data from the MATLAB file
        emg_data = mat_contents['SIG']

        # convert from mV to µV
        emg_data = emg_data / EMGGainFactor

        # create folder for the results in dataset/Alessandro_sEMG/AINN_DECOMP_PROX_35-50-70_FINISHED_TO/PRE
        dataset_path = 'dataset' + '/test/'
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

        plt.figure()
        plt.plot(emg_hr_reshaped[0, 0])
        plt.savefig('emg_hr_reshaped.png')
        plt.close()

        # Reshape arrays to channel x samples, counting channels from top to bottom
        # emg_lr_reshaped_final = np.flipud(emg_lr_reshaped).transpose(1, 0, 2).reshape(-1, emg_lr_reshaped.shape[-1])
        # emg_hr_reshaped_final = np.flipud(emg_hr_reshaped).transpose(1, 0, 2).reshape(-1, emg_hr_reshaped.shape[-1])

        windows = emg_length - 2048

        # window the signal to 2048 samples
        # 51 samples equals 25ms which is approximately 1 action potential --> 2AP overlapping
        for window in range(0, windows, (2048 - 102)):
            print("window", window)
            # The length of one sample is 2048 which is equal to 1 second
            emg_lr_windowed = emg_lr_reshaped[:, :, window:(window + 2048)]
            emg_hr_windowed = emg_hr_reshaped[:, :, window:(window + 2048)]

            # transpose for samples x channels shape
            # emg_lr_final = emg_lr_windowed.transpose()
            # emg_hr_final = emg_hr_windowed.transpose()

            # save high resolution data and add file name to list
            hr_file_location = dataset_path + f'path{i}-window{window}-hr.npy'
            np.save(hr_file_location, emg_hr_windowed)
            hr_filename = f'path{i}-window{window}-hr.npy'
            hr_filenames_test.append(hr_filename)

            # save low resolution data and add file name to list
            lr_file_location = dataset_path + f'path{i}-window{window}-lr.npy'
            np.save(lr_file_location, emg_lr_windowed)
            lr_filename = f'path{i}-window{window}-lr.npy'
            lr_filenames_test.append(lr_filename)


    # Combine LR and HR filenames row-wise
    data = zip(lr_filenames_test, hr_filenames_test)
    # File path for the CSV file
    csv_file = 'data.csv'

    # Open the CSV file in append mode and write the data
    with open('dataset/test' + '/' + csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Add a header if the file is empty
        if file.tell() == 0:
            writer.writerow(['lr_filename;hr_filename'])

        # Add each LR/HR pair as a row in the CSV file
        for lr_filename_test, hr_filename_test in data:
            writer.writerow([lr_filename_test + ';' + hr_filename_test])


def create_tar_from_folder(folder_path, tar_filename):
    with tarfile.open(tar_filename, 'w') as tar:
        # Walk through the folder and its subfolders
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Add each file to the tar file
                tar.add(os.path.join(root, file))


if __name__ == '__main__':
    # data_paths = ['data/Dominik/offline/20230720-085154/Ambra_20230720-085154_Follow Trajectory.pkl']
    #               #'data/Dominik/offline/20230720-090835/Ambra_20230720-090835_Follow Trajectory.pkl',
    #               #'data/Dominik/offline/20230720-091107/Ambra_20230720-091107_Follow Trajectory.pkl']
    # # data_paths = ['data/Dominik/offline/20230720-091659/Ambra_20230720-091659_Follow Trajectory.pkl']
    #
    # movement_name = 'thumb_slow'
    #
    # dataset_creator_dominik(data_paths)

    # TODO: use 10 subjects for the dataset and 2 subjects for the testset
    train_paths = ['data/Alessandros Daten sEMG/Train and Val/AINN_DECOMP_PROX_35-50-70_FINISHED_TO/POST/400InAnfd170619145709 Rampa_35MVC_5MVCs_1acontrazione_TODECOMPOSE_offset0_length29.682_runs250_EDITED_MATTEO.mat',
                   'data/Alessandros Daten sEMG/Train and Val/AINN_DECOMP_PROX_35-50-70_FINISHED_TO/PRE/400InAnfd170605141107 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length27_runs250_FINISHED_ANDREA.mat',
                   'data/Alessandros Daten sEMG/Train and Val/ESAN_DECOMP_PROX_35-50-70_FINISHED_TO/POST/400SaEnRo171205164509 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length30_runs100_FINISHED.mat',
                   'data/Alessandros Daten sEMG/Train and Val/ESAN_DECOMP_PROX_35-50-70_FINISHED_TO/PRE/400SaEnRo171128171409 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length30_runs250_FINISHED_ANDREA.mat',
                   'data/Alessandros Daten sEMG/Train and Val/LRUM_DECOMP_PROX_35-50-70_FINISHED/POST/400RuLoRo171026112715 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs100_EDITED_MATTEO.mat',
                   'data/Alessandros Daten sEMG/Train and Val/LRUM_DECOMP_PROX_35-50-70_FINISHED/PRE/400RuLoRo171019122211 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length29.993_runs250_FINISHED.mat',
                   'data/Alessandros Daten sEMG/Train and Val/LSEB_DECOMP_PROX_35-50-70_FINISHED/POST/400SeLuRo171212105809 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length28.224_runs100_FINISHED.mat',
                   'data/Alessandros Daten sEMG/Train and Val/LSEB_DECOMP_PROX_35-50-70_FINISHED/PRE/400SeLuRo171206114409 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs10_FINISHED.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MCAC_DECOMP_PROX_35-50-70_FINISHED/POST/400CaMARo171027153608 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs100_EDITED_MATTEO.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MCAC_DECOMP_PROX_35-50-70_FINISHED/PRE/400CaMARo171017190803 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length28.602_runs250_FINISHED.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MCIAN_DECOMP_PROX_35-50-70_FINISHED/POST/400CiMaRo180116151710 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs100_EDITED_MATTEO.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MCIAN_DECOMP_PROX_35-50-70_FINISHED/PRE/400CiMaRo171222104608 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs100_FINISHED_ANDREA.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MGIR_DECOMP_PROX_35-50-70_FINISHED/POST/400GiMiRo171207102708 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30.212_runs100_FINISHED.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MGIR_DECOMP_PROX_35-50-70_FINISHED/PRE/400GiMiRo171130110703 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length28.771_FINISHED.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MIZZO_DECOMP_PROX_35-50-70_FINISHED/POST/400IzMaRo171019160209 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length30_runs100_EDITED_MATTEO.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MIZZO_DECOMP_PROX_35-50-70_FINISHED/PRE/400IzMaRo171012113009 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs250_EDITED_MATTEO.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MMAGN_DECOMP_PROX_35-50-70_FINISHED/POST/400MaMiRo171130170007 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length29.317_runs100_FINISHED.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MMAGN_DECOMP_PROX_35-50-70_FINISHED/PRE/400MaMiRo171123105508 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs100_FINISHED.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MMAND_DECOMP_PROX_35_50_70_FINISHED/POST/400MaMaRo180115175709 Rampa_35MVC_5MVCs_1_TODECOMPOSE_offset0_length30_runs100_EDITED_MATTEO.mat',
                   'data/Alessandros Daten sEMG/Train and Val/MMAND_DECOMP_PROX_35_50_70_FINISHED/PRE/400MaMaRo171220163607 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs100_FINISHED_MATTEO.mat']

    test_paths = ['data/Alessandros Daten sEMG/Test/RPAR_DECOMP_PROX_35-50-70_FINISHED/POST/400PaRiRo170628134308 Rampa_35MVC_5MVCs_2acontrazione_TODECOMPOSE_offset0_length30_runs250_EDITED_MATTEO.mat',
                  'data/Alessandros Daten sEMG/Test/RPAR_DECOMP_PROX_35-50-70_FINISHED/PRE/400PaRifd170606133208 Rampa_35MVC_5MVCs_2acontrazione_TODECOMPOSE_offset0_length27_runs250_FINISHED_MATTEO.mat',
                  'data/Alessandros Daten sEMG/Test/SNUC_DECOMP_PROX_35-50-70_FINISHED/POST/400NuStRo171221103923 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length29.665_runs100_FINISHED.mat',
                  'data/Alessandros Daten sEMG/Test/SNUC_DECOMP_PROX_35-50-70_FINISHED/PRE/400NuStRo171215104010 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs100_FINISHED.mat']

    dataset_creator_alessandro_13x5(train_paths, test_paths)

    # Define folder path and tar filename
    folder_path = 'dataset'
    tar_filename = 'dataset.tar'

    create_tar_from_folder(folder_path, tar_filename)
