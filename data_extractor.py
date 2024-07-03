"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains functions to open and plot EMG data and to create a dataset from
                    the EMG data including a csv file with their directories. The dataset_creator_alessandro_13x5
                    function creates a dataset from the EMG data with shape 13x5. It creates a folder for the training and
                    test data and saves the data in the folder. It also creates a csv file with the directories of the data.

You can run this code by defining the data paths of the raw data and calling the dataset_creator_dominik or
dataset_creator_alessandro function with the data paths as an argument.
"""

import json
import os
import pickle
import tarfile
import csv
import matplotlib
import numpy as np
from scipy.io import loadmat

matplotlib.use('Agg')  # Use the 'Agg' backend
import matplotlib.pyplot as plt


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

def dataset_creator_alessandro_13x5(train_paths: list[str], test_paths: list[str], dataset_name: str, sequence_length: int) -> None:
    """
    This function creates a dataset from the EMG data with shape 13x5. It creates a folder for the training and test data
    and saves the data in the folder. It also creates a csv file with the directories of the data.

    Parameters
    ----------
    train_paths: list[str]
                    list of paths to the training data
    test_paths: list[str]
                    list of paths to the test data
    dataset_name: str
                    name of the dataset
    sequence_length: int
                    target length of the sEMG sequence

    Returns
    -------
    None
    """

    ######################## Training ########################

    EMGGainFactor = 5 / (2 ** 16) / 150 * 1000

    # List of LR and HR filenames (replace this with your filenames)
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
        dataset_path = dataset_name + '/train/'
        print(dataset_path)

        # Create the folder if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)

        emg_length = emg_data[0][0][0].shape[0]
        emg_hr_reshaped = np.zeros((emg_data.shape[0], emg_data.shape[1], emg_length))

        # Reshape HR data into 3D
        for row in range(emg_data.shape[0]):
            for col in range(emg_data.shape[1]):
                if (row == (emg_data.shape[0] - 1) and col == 0):
                    continue
                else:
                    sub_array = emg_data[row][col]
                    emg_hr_reshaped[row, col] = sub_array

        plt.figure()
        plt.plot(emg_hr_reshaped[0, 0])
        plt.savefig('emg_hr_reshaped.png')
        plt.close()

        if i == 0:
            combined_data = emg_hr_reshaped.flatten()
        else:
            combined_data = np.concatenate((combined_data, emg_hr_reshaped.flatten()))

        windows = emg_length - sequence_length

        # window the signal to sequence_length samples
        # 51 samples equals 25ms which is approximately 1 action potential
        for window in range(0, windows, (sequence_length - 51)):
            print("window", window)
            # The length of one sample is sequence_length which is equal to 10 second
            emg_hr_windowed = emg_hr_reshaped[:, :, window:(window + sequence_length)]

            # save high resolution data and add file name to list
            hr_file_location = dataset_path + f'path{i}-window{window}-hr.npy'
            np.save(hr_file_location, emg_hr_windowed)
            hr_filename = f'path{i}-window{window}-hr.npy'
            hr_filenames_training.append(hr_filename)

    # Get mean and std for standardization
    mean = np.mean(combined_data)
    std = np.std(combined_data)

    # Create a dictionary for the hyperparameters
    mean_std_dict = {
        "mean": mean,
        "std": std,
    }

    with open(dataset_name + '/train' + '/mean_std.json', 'w') as file:
        json.dump(mean_std_dict, file)

    # File path for the CSV file
    csv_file = 'data.csv'

    # Open the CSV file in append mode and write the data
    with open(dataset_name + '/train' + '/' + csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Add a header if the file is empty
        if file.tell() == 0:
            writer.writerow(['hr_filename'])

        # Add each HR filename as a row in the CSV file
        for hr_filename_training in hr_filenames_training:
            writer.writerow([hr_filename_training])

    ######################## Test ########################

    # List of HR filenames (replace this with your filenames)
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

        # create folder for the results
        dataset_path = dataset_name + '/test/'
        print(dataset_path)

        # Create the folder if it doesn't exist
        os.makedirs(dataset_path, exist_ok=True)

        emg_length = emg_data[0][0][0].shape[0]
        emg_hr_reshaped = np.zeros((emg_data.shape[0], emg_data.shape[1], emg_length))

        # Reshape HR data into 3D
        for row in range(emg_data.shape[0]):
            for col in range(emg_data.shape[1]):
                if (row == (emg_data.shape[0] - 1) and col == 0):
                    continue
                else:
                    sub_array = emg_data[row][col]
                    emg_hr_reshaped[row, col] = sub_array

        # In case that last dimension is not divisible by 64 delete samples so that skip connection between contracting
        # and expanding path is possible
        if emg_hr_reshaped.shape[2] % 64 != 0:
            emg_hr_reshaped = emg_hr_reshaped[:, :, :-(emg_hr_reshaped.shape[2] % 64)]

        plt.figure()
        plt.plot(emg_hr_reshaped[0, 0])
        plt.savefig('emg_hr_reshaped.png')
        plt.close()

        # save high resolution data and add file name to list
        hr_file_location = dataset_path + f'path{i}-hr.npy'
        np.save(hr_file_location, emg_hr_reshaped)
        hr_filename = f'path{i}-hr.npy'
        hr_filenames_test.append(hr_filename)

    # File path for the CSV file
    csv_file = 'data.csv'

    # Open the CSV file in append mode and write the data
    with open(dataset_name + '/test' + '/' + csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Add a header if the file is empty
        if file.tell() == 0:
            writer.writerow(['hr_filename'])

        # Add each HR filename as a row in the CSV file
        for hr_filename_test in hr_filenames_test:
            writer.writerow([hr_filename_test])


def create_tar_from_folder(folder_path: str) -> None:
    """
    This function creates a tar file from a folder.

    Parameters
    ----------
    folder_path: str
                    path to the folder

    Returns
    -------
    None

    """
    tar_filename = folder_path + '.tar'

    with tarfile.open(tar_filename, 'w') as tar:
        # Walk through the folder and its subfolders
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                # Add each file to the tar file
                tar.add(os.path.join(root, file))


if __name__ == '__main__':

    # use 10 subjects for the dataset and 2 subjects for the testset
    train_paths = [
        'data/Alessandros Daten sEMG/Train and Val/AINN_DECOMP_PROX_35-50-70_FINISHED_TO/POST/400InAnfd170619145709 Rampa_35MVC_5MVCs_1acontrazione_TODECOMPOSE_offset0_length29.682_runs250_EDITED_MATTEO.mat',
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

    test_paths = [
        'data/Alessandros Daten sEMG/Test/RPAR_DECOMP_PROX_35-50-70_FINISHED/POST/400PaRiRo170628134308 Rampa_35MVC_5MVCs_2acontrazione_TODECOMPOSE_offset0_length30_runs250_EDITED_MATTEO.mat',
        'data/Alessandros Daten sEMG/Test/RPAR_DECOMP_PROX_35-50-70_FINISHED/PRE/400PaRifd170606133208 Rampa_35MVC_5MVCs_2acontrazione_TODECOMPOSE_offset0_length27_runs250_FINISHED_MATTEO.mat',
        'data/Alessandros Daten sEMG/Test/SNUC_DECOMP_PROX_35-50-70_FINISHED/POST/400NuStRo171221103923 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length29.665_runs100_FINISHED.mat',
        'data/Alessandros Daten sEMG/Test/SNUC_DECOMP_PROX_35-50-70_FINISHED/PRE/400NuStRo171215104010 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs100_FINISHED.mat']

    # Dataset UNet variants
    dataset_creator_alessandro_13x5(train_paths, test_paths, 'dataset', sequence_length=2048)

    #Define folder path and tar filename
    folder_path = 'dataset'

    create_tar_from_folder(folder_path)


    # Dataset SRGAN
    dataset_creator_alessandro_13x5(train_paths, test_paths, 'dataset_512', sequence_length=512)

    # Define folder path and tar filename
    folder_path = 'dataset_512'

    create_tar_from_folder(folder_path)

    # Dataset LSTM
    dataset_creator_alessandro_13x5(train_paths, test_paths, 'dataset_LSTM', sequence_length=40960)

    # Define folder path and tar filename
    folder_path = 'dataset_LSTM'

    create_tar_from_folder(folder_path)


