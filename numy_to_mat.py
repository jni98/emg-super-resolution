import os

import scipy.io

import numpy as np


def numpy_to_mat(file, file_path_mat, shape, whiten):
    # Create a NumPy array filled with None to represent the cell variable
    cell_variable = np.empty(shape, dtype=object)

    # Load corresbonding mat file
    mat = scipy.io.loadmat(file_path_mat)

    mat_ref_signal = mat['ref_signal']

    # Load numpy file to be converted
    emg = np.load('results/' + file + '.npy')

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

    # Fill the cell_variable with the data from the NumPy array
    for i in range(shape[0]):
        for j in range(shape[1]):
            cell = emg[i, j]
            # Adding a new dimension at the beginning using np.expand_dims
            cell_new = np.expand_dims(cell, axis=0)
            cell_variable[i, j] = cell_new

    fsamp = 2048.0

    # Save the numpy array to a .mat file
    scipy.io.savemat('results_mat/' + file + '.mat',
                     {'SIG': cell_variable, 'fsamp': np.float64(fsamp), 'ref_signal': mat_ref_signal})


if __name__ == '__main__':
    files_interpolation = ['linear_Interpolation/path0-lr', 'linear_Interpolation/path1-lr',
                           'linear_Interpolation/path2-lr', 'linear_Interpolation/path3-lr']
    files_groundtruth = ['groundtruth/path0-hr', 'groundtruth/path1-hr', 'groundtruth/path2-hr', 'groundtruth/path3-hr']
    files_groundtruth_lr = ['groundtruth/path0-lr', 'groundtruth/path1-lr', 'groundtruth/path2-lr', 'groundtruth/path3-lr']

    files_model2 = ['model_2/path0-lr', 'model_2/path1-lr', 'model_2/path2-lr', 'model_2/path3-lr']
    files_model1 = ['model_1/path0-lr', 'model_1/path1-lr', 'model_1/path2-lr', 'model_1/path3-lr']
    files_model2_Fourier = ['model_2_Fourier/path0-lr', 'model_2_Fourier/path1-lr', 'model_2_Fourier/path2-lr', 'model_2_Fourier/path3-lr']
    files_model2_Fourier_Layer = ['model_2_Fourier_Layer/path0-lr', 'model_2_Fourier_Layer/path1-lr', 'model_2_Fourier_Layer/path2-lr', 'model_2_Fourier_Layer/path3-lr']
    files_model2_Wavelet_Layer= ['model_2_Wavelet_Layer/path0-lr', 'model_2_Wavelet_Layer/path1-lr', 'model_2_Wavelet_Layer/path2-lr', 'model_2_Wavelet_Layer/path3-lr']

    test_paths_mat = [
        'data/Alessandros Daten sEMG/Test/RPAR_DECOMP_PROX_35-50-70_FINISHED/POST/400PaRiRo170628134308 Rampa_35MVC_5MVCs_2acontrazione_TODECOMPOSE_offset0_length30_runs250_EDITED_MATTEO.mat',
        'data/Alessandros Daten sEMG/Test/RPAR_DECOMP_PROX_35-50-70_FINISHED/PRE/400PaRifd170606133208 Rampa_35MVC_5MVCs_2acontrazione_TODECOMPOSE_offset0_length27_runs250_FINISHED_MATTEO.mat',
        'data/Alessandros Daten sEMG/Test/SNUC_DECOMP_PROX_35-50-70_FINISHED/POST/400NuStRo171221103923 Rampa_35MVC_5MVCs_1a_TODECOMPOSE_offset0_length29.665_runs100_FINISHED.mat',
        'data/Alessandros Daten sEMG/Test/SNUC_DECOMP_PROX_35-50-70_FINISHED/PRE/400NuStRo171215104010 Rampa_35MVC_5MVCs_2a_TODECOMPOSE_offset0_length30_runs100_FINISHED.mat']

    os.makedirs('results_mat/linear_Interpolation', exist_ok=True)
    os.makedirs('results_mat/groundtruth', exist_ok=True)
    os.makedirs('results_mat/model_2', exist_ok=True)
    os.makedirs('results_mat/model_2_Fourier_Layer', exist_ok=True)
    os.makedirs('results_mat/model_2_Wavelet_Layer', exist_ok=True)

    # for file, test_path_mat in zip(files_interpolation, test_paths_mat):
    #     numpy_to_mat(file, test_path_mat, (13,5))
    #     print(file)
    #     print("Done")
    #
    # for file, test_path_mat in zip(files_groundtruth, test_paths_mat):
    #     numpy_to_mat(file, test_path_mat,(13,5))
    #     print(file)
    #     print("Done")
    #
    # for file, test_path_mat in zip(files_model2, test_paths_mat):
    #     numpy_to_mat(file, test_path_mat,(13,5))
    #     print(file)
    #     print("Done")
    #
    # for file, test_path_mat in zip(files_model1, test_paths_mat):
    #     numpy_to_mat(file, test_path_mat,(13,5))
    #     print(file)
    #     print("Done")

    # for file, test_path_mat in zip(files_model2_Fourier, test_paths_mat):
    #     numpy_to_mat(file, test_path_mat,(13,5))
    #     print(file)
    #     print("Done")

    for file, test_path_mat in zip(files_model2_Fourier_Layer, test_paths_mat):
        numpy_to_mat(file, test_path_mat, (13,5), whiten =True)
        print(file)
        print("Done")

    for file, test_path_mat in zip(files_model2_Wavelet_Layer, test_paths_mat):
        numpy_to_mat(file, test_path_mat, (13,5), whiten =True)
        print(file)
        print("Done")