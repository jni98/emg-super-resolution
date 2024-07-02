import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, filtfilt


def destandardize_data(standardized_data, mean, std):
    """
    Destandardizes the data using the provided mean and standard deviation.

    Parameters
    ----------
    standardized_data: torch.Tensor
        The standardized data to be destandardized
    mean: float
        The mean used for standardization
    std: float
        The standard deviation used for standardization

    Returns
    -------
    torch.Tensor
        The destandardized data
    """
    return standardized_data * std + mean


def standardization(data: np.array, mean: float, std: float) -> np.array:
    """
    Standardize the data using the mean and std values
    Parameters
    ----------
    data: np.array
        Data to be standardized
    mean: float
        Mean value of the complete dataset
    std: float
        Standard deviation value of the complete dataset

    Returns
    -------
    standardized_data: np.array
        Standardized data
    """

    # Min-max normalization using precalculated min and max values
    standardized_data = (data - mean) / std
    return standardized_data


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    """
    Apply a bandpass filter to the data.

    Parameters
    ----------
    data: np.array
        Data to be filtered
    lowcut: float
        Lower cut-off frequency
    highcut: float
        Higher cut-off frequency
    fs: int
        Sampling frequency
    order: int
        Order of the filter

    Returns
    -------
    np.array
        Filtered data
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, data)

    return filtered_signal


def grid_interpolation( lr_emg: np.ndarray, hr_emg_shape: tuple, method: str) -> np.ndarray:
    """
    Performs cubic interpolation on the low resolution data to get high resolution data.
    Parameters
    ----------
    lr_emg: numpy.ndarray
            Low resolution emg data.
    hr_emg_shape: tuple
            Shape of the high resolution emg data.
    method: str
            Method of interpolation. On default set to linear.

    Returns
    -------
    interpolated_signal: numpy.ndarray
            High resolution emg data.
    """
    # Create a grid for the original signal
    x = np.linspace(0, 1, lr_emg.shape[0])
    y = np.linspace(0, 1, lr_emg.shape[1])
    z = np.linspace(0, 1, lr_emg.shape[2])

    # Create a new grid for the desired interpolated signal
    new_x = np.linspace(0, 1, hr_emg_shape[0])
    new_y = np.linspace(0, 1, hr_emg_shape[1])
    new_z = np.linspace(0, 1, hr_emg_shape[2])

    # Create meshgrid from new grid
    mesh_x, mesh_y, mesh_z = np.meshgrid(new_x, new_y, new_z, indexing='ij')

    # Create interpolator
    interpolator = RegularGridInterpolator((x, y, z), lr_emg, method=method)
    # print("Created the interpolater")

    # Interpolate
    interpolated_signal = interpolator((mesh_x, mesh_y, mesh_z))
    # print("Finished the interpolator")
    return interpolated_signal
