import os
import scipy.io
import numpy as np
from scipy.signal import butter, filtfilt


def numpy_to_mat(file, file_path_mat, shape, whiten=False, file_groundtruth=None, gaussian_noise=False, noise_std=50e-6,
                 lr=False):
    # Create a NumPy array filled with None to represent the cell variable
    cell_variable = np.empty(shape, dtype=object)

    fsamp = 2048.0

    # Load corresponding mat file
    mat = scipy.io.loadmat(file_path_mat)
    mat_ref_signal = mat['ref_signal']

    # Load numpy file to be converted
    emg = np.load('results/' + file + '.npy')
    emg = emg.squeeze()

    if whiten:
        # Assuming emg_data is your EMG data of shape (13, 5, 60000)
        emg_data_shape = emg.shape

        # Reshape the data to 2D
        emg_data_2d = np.reshape(emg, (np.prod(emg_data_shape[:2]), emg_data_shape[2]))

        # Compute the covariance matrix
        cov_matrix = np.cov(emg_data_2d)

        # Compute the whitening transformation
        _, S, V = np.linalg.svd(cov_matrix)
        whitening_matrix = np.dot(np.dot(V.T, np.diag(1.0 / np.sqrt(S + 1e-5))), V)

        # Apply the whitening transformation
        whitened_data_2d = np.dot(whitening_matrix, emg_data_2d)

        # Reshape the whitened data back to its original shape
        emg = np.reshape(whitened_data_2d, emg_data_shape)

    if gaussian_noise:
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, emg.shape)
        emg_noise = emg + noise

        # Fill the cell_variable with the data from the NumPy array
        for i in range(shape[0]):
            for j in range(shape[1]):
                cell = emg_noise[i, j]
                # Adding a new dimension at the beginning using np.expand_dims
                cell_new = np.expand_dims(cell, axis=0)
                cell_variable[i, j] = cell_new

        # Save the numpy array to a .mat file
        scipy.io.savemat('results_mat/' + file + '_noise' + '.mat',
                         {'SIG': cell_variable, 'fsamp': np.float64(fsamp), 'ref_signal': mat_ref_signal})

    if file_groundtruth != None:
        # Load ground truth data
        groundtruth = np.load('results/' + file_groundtruth + '.npy')
        groundtruth = groundtruth.squeeze()
        groundtruth_filtered = apply_bandpass_filter(groundtruth)

        # Create combined cell variable
        cell_variable_combined = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                if i % 2 == 0 and j % 2 == 0:
                    cell = groundtruth_filtered[i, j]
                else:
                    cell = emg[i, j]
                cell_new = np.expand_dims(cell, axis=0)
                cell_variable_combined[i, j] = cell_new

        # Save the combined numpy array to a .mat file
        scipy.io.savemat('results_mat/' + file + '_combined' + '.mat',
                         {'SIG': cell_variable_combined, 'fsamp': np.float64(fsamp), 'ref_signal': mat_ref_signal})

    # Fill the cell_variable with the data from the NumPy array
    for i in range(shape[0]):
        for j in range(shape[1]):
            cell = emg[i, j]
            # Adding a new dimension at the beginning using np.expand_dims
            cell_new = np.expand_dims(cell, axis=0)
            cell_variable[i, j] = cell_new

    # Save the numpy array to a .mat file
    scipy.io.savemat('results_mat/' + file + '.mat',
                         {'SIG': cell_variable, 'fsamp': np.float64(fsamp), 'ref_signal': mat_ref_signal})

    if lr:
        emg_lr = emg[::2, ::2]
        cell_variable_lr = np.empty((emg_lr.shape[0], emg_lr.shape[1]), dtype=object)
        for i in range(emg_lr.shape[0]):
            for j in range(emg_lr.shape[1]):
                cell = emg_lr[i, j]
                # Adding a new dimension at the beginning using np.expand_dims
                cell_new = np.expand_dims(cell, axis=0)
                cell_variable_lr[i, j] = cell_new

        # Save the numpy array to a .mat file
        scipy.io.savemat('results_mat/' + file + '_lr' + '.mat',
                         {'SIG': cell_variable_lr, 'fsamp': np.float64(fsamp), 'ref_signal': mat_ref_signal})

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


if __name__ == '__main__':

    # Add files that need to be converted to the corresponding folder
    files_interpolation = ['linear_Interpolation/path0-lr', 'linear_Interpolation/path1-lr',
                           'linear_Interpolation/path2-lr', 'linear_Interpolation/path3-lr']
    files_bicubic_interpolation = ['cubic_interpolation/path0-lr', 'cubic_interpolation/path1-lr',
                                   'cubic_interpolation/path2-lr', 'cubic_interpolation/path3-lr']
    files_groundtruth = ['groundtruth/path0-hr', 'groundtruth/path1-hr', 'groundtruth/path2-hr', 'groundtruth/path3-hr',
                         'groundtruth/SIG-hr']
    files_groundtruth_lr = ['groundtruth/path0-lr', 'groundtruth/path1-lr', 'groundtruth/path2-lr',
                            'groundtruth/path3-lr']

    files_model2 = ['model_2/path0-lr', 'model_2/path1-lr', 'model_2/path2-lr', 'model_2/path3-lr']
    files_model1 = ['model_1/path0-lr', 'model_1/path1-lr', 'model_1/path2-lr', 'model_1/path3-lr']
    files_model2_Fourier = ['model_2_Fourier/path0-lr', 'model_2_Fourier/path1-lr', 'model_2_Fourier/path2-lr',
                            'model_2_Fourier/path3-lr']
    files_model2_Wavelet = ['model_2_Wavelet/path0-lr', 'model_2_Wavelet/path1-lr', 'model_2_Wavelet/path2-lr',
                            'model_2_Wavelet/path3-lr']
    files_model3_LSTM = ['model_3_LSTM/path0-sr', 'model_3_LSTM/path1-sr', 'model_3_LSTM/path2-sr',
                         'model_3_LSTM/path3-sr']  # , 'model_3_LSTM/SIG-sr']

    files_model2_Fourier_hrsr = ['model_2_Fourier/hr-sr/path0-lr', 'model_2_Fourier/hr-sr/path1-lr', 'model_2_Fourier/hr-sr/path2-lr',
                            'model_2_Fourier/hr-sr/path3-lr']
    files_model2_Wavelet_hrsr = ['model_2_Wavelet/hr-sr/path0-lr', 'model_2_Wavelet/hr-sr/path1-lr', 'model_2_Wavelet/hr-sr/path2-lr',
                            'model_2_Wavelet/hr-sr/path3-lr']

    # Corresponding groundtruth matlab files to add ref signal to the synthetic signals
    test_paths_mat = [
        'data/Alessandros Daten sEMG/Test/RPAR_DECOMP_PROX_35-50-70_FINISHED/POST/400PaRiRo170628134308 Rampa_35MVC_5MVCs_2acontrazione_TODECOMPOSE_offset0_length30_runs250_EDITED_MATTEO.mat',
        'data/Alessandros Daten sEMG/Test/RPAR_DECOMP_PROX_35-50-70_FINISHED/PRE/400PaRifd170606133208 Rampa_35MVC_5MVCs_2acontrazione_TODECOMPOSE_offset0_length27_runs250_FINISHED_MATTEO.mat',
        'data/Alessandros Daten sEMG/Test/SNUC_DECOMP_PROX_35-50-70_FINISHED/POST/400NuStRo171221103923 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length29.665_runs100_FINISHED.mat',
        'data/Alessandros Daten sEMG/Test/SNUC_DECOMP_PROX_35-50-70_FINISHED/PRE/400NuStRo171215104010 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs100_FINISHED.mat',
        'data/13x5-SIG.mat']

    os.makedirs('results_mat/linear_Interpolation', exist_ok=True)
    os.makedirs('results_mat/groundtruth_hr', exist_ok=True)
    os.makedirs('results_mat/groundtruth_lr', exist_ok=True)
    os.makedirs('results_mat/cubic_interpolation', exist_ok=True)
    os.makedirs('results_mat/model_1', exist_ok=True)
    os.makedirs('results_mat/model_2', exist_ok=True)
    os.makedirs('results_mat/model_2_Fourier', exist_ok=True)
    os.makedirs('results_mat/model_2_Wavelet', exist_ok=True)
    os.makedirs('results_mat/model_3_LSTM', exist_ok=True)

    # groundtruth
    for file, test_path_mat in zip(files_groundtruth, test_paths_mat):
        numpy_to_mat(file, test_path_mat,(13, 5), lr = True)
        print(file)
        print("Done")

    # linear
    for file, test_path_mat in zip(files_interpolation, test_paths_mat):
        numpy_to_mat(file, test_path_mat, (13, 5), whiten=False, gaussian_noise=True)
        print(file)
        print("Done")

    # bicubic
    for file, test_path_mat in zip(files_bicubic_interpolation, test_paths_mat):
        numpy_to_mat(file, test_path_mat, (13, 5), whiten=False, gaussian_noise=True)
        print(file)
        print("Done")

    # model 1
    for file, test_path_mat, file_gr in zip(files_model1, test_paths_mat, files_groundtruth):
        numpy_to_mat(file, test_path_mat, (13, 5),file_groundtruth=file_gr, whiten=False, gaussian_noise=True)
        print(file)
        print("Done")

    # model 2
    for file, test_path_mat, file_gr in zip(files_model2, test_paths_mat, files_groundtruth):
        numpy_to_mat(file, test_path_mat, (13, 5),file_groundtruth=file_gr, whiten=False, gaussian_noise=True)
        print(file)
        print("Done")

    # model 2 Fourier
    for file, test_path_mat, file_gr in zip(files_model2_Fourier, test_paths_mat, files_groundtruth):
        numpy_to_mat(file, test_path_mat, (13, 5),file_groundtruth=file_gr, whiten=False, gaussian_noise=True)
        print(file)
        print("Done")

    # model 2 wavelet
    for file, test_path_mat, file_gr in zip(files_model2_Wavelet, test_paths_mat, files_groundtruth):
        numpy_to_mat(file, test_path_mat, (13, 5), file_groundtruth=file_gr, whiten=False, gaussian_noise=True)
        print(file)
        print("Done")

    # model 3
    for file, test_path_mat, file_gr in zip(files_model3_LSTM, test_paths_mat, files_groundtruth):
        numpy_to_mat(file, test_path_mat,  (13, 5),file_groundtruth=file_gr, whiten=False, gaussian_noise=True)
        print(file)
        print("Done")

