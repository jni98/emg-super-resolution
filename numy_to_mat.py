import os

import scipy.io

import numpy as np


def numpy_to_mat(file):
    # Create a NumPy array filled with None to represent the cell variable
    cell_variable = np.empty((13, 5), dtype=object)

    gain_factor = 5 / (2 ** 16) / 150 * 1000

    emg = np.load('results/' + file + '.npy')
    emg = emg * gain_factor

    # Fill the cell_variable with the data from the NumPy array
    for i in range(13):
        for j in range(5):
            cell = emg[i, j]
            # Adding a new dimension at the beginning using np.expand_dims
            cell_new = np.expand_dims(cell, axis=0)
            cell_variable[i, j] = cell_new

    # Save the numpy array to a .mat file
    scipy.io.savemat('results_mat/' + file + '.mat', {'SIG': cell_variable, 'fsamp': 2048})
    print("Hello")


if __name__ == '__main__':
    files = ['linear_Interpolation/path0-lr', 'linear_Interpolation/path1-lr',
             'linear_Interpolation/path2-lr', 'linear_Interpolation/path3-lr',
             'groundtruth/path0-hr', 'groundtruth/path1-hr',
             'groundtruth/path2-hr', 'groundtruth/path3-hr']
    # ,'model2/path0-lr.npy', 'model2/path1-lr.npy', 'model2/path2-lr.npy', 'model2/path3-lr.npy']

    os.makedirs('results_mat/linear_Interpolation', exist_ok=True)
    os.makedirs('results_mat/groundtruth', exist_ok=True)
    os.makedirs('results_mat/model2', exist_ok=True)


    for file in files:
        numpy_to_mat(file)
        print(file)
        print("Done")
