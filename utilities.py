import math

import numpy as np

def data_13x5_2D_to_3D(data: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Reshape the 2D data to 3D grid shape
    Parameters
    ----------
    data: np.ndarray
        2D data
    rows: int
        Number of rows
    cols: int
        Number of columns

    Returns
    -------
    data_grid: np.ndarray
        3D grid shaped data
    """
    # reshape back to grid shape
    data = data.squeeze()
    data = data.transpose()

    data_reshaped = data.reshape(cols, rows, data.shape[1])

    # Transpose and flip to get to correct grid numbering
    data_transposed = np.transpose(data_reshaped, axes=(1, 0, 2))
    data_grid = np.flipud(data_transposed)

    # reverse every second row
    data_grid[:, 1::2] = data_grid[::-1, 1::2]

    return data_grid

def data_13x5_3D_to_2D(data: np.ndarray) -> np.ndarray:
    """
    Reshape the 3D grid data to 2D shape
    Parameters
    ----------
    data: np.ndarray
        3D data

    Returns
    -------
    data_2D: np.ndarray
        2D grid shaped data
    """

    # Reshape arrays to channel x samples, counting channels from top to bottom
    data_2D = np.flipud(data).transpose(1, 0, 2).reshape(-1, data.shape[-1])

    data_2D = data_2D.transpose()

    return data_2D


def data_8x8_2D_to_3D(data: np.ndarray) -> np.ndarray:
    """
    Reshape the 2D data to 3D grid shape
    Parameters
    ----------
    data: np.ndarray
        2D data
    shape: tuple
        Shape of the 3D grid

    Returns
    -------
    data_grid: np.ndarray
        3D grid shaped data
    """

    data = data.squeeze().numpy()
    data = data.transpose()
    data_reshaped = data.reshape(math.sqrt(data.shape[0]), math.sqrt(data.shape[0]), data.shape[1])

    # Transpose and flip to get to correct grid numbering
    data_transposed = np.transpose(data_reshaped, axes=(1, 0, 2))
    data_grid = np.flipud(data_transposed)

    return data_grid

def data_8x8_3D_to_2D(data: np.ndarray) -> np.ndarray:
    """
    Reshape the 3D grid data to 2D shape
    Parameters
    ----------
    data: np.ndarray
        3D data

    Returns
    -------
    data_2D: np.ndarray
        2D grid shaped data
    """

    data_2D = np.flipud(data).transpose(1, 0, 2).reshape(-1, data.shape[-1])

    data_2D = data_2D.transpose()

    return data_2D

