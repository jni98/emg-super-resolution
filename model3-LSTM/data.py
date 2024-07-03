import json

import torch
import numpy as np
import pytorch_lightning as pl
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from emg_augmentations import GaussianNoise, WaveletDecomposition, MagnitudeWarping
import random


class TimeseriesDataset(Dataset):
    '''
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    '''
    def __init__(self, X: np.ndarray,directory: str,mean, std, transform=None, augment = False):
        self.X = X
        self.directory = directory
        self.transform = transform
        self.mean = mean
        self.std = std
        self.augment = augment

        self.gaussian_noise_generator = GaussianNoise(target_snr__db=5.0)
        self.wavelet_decomposition = WaveletDecomposition(b=0.25, wavelet="db7", level=5, nr_of_grids=1)
        self.magnitude_warping = MagnitudeWarping(nr_of_grids=1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple:
        if torch.is_tensor(index):
            index = index.tolist()

        # Load file paths
        hr_emg_path = self.X.iloc[index]['hr_filename']

        # Load and preprocess HR data
        hr_emg = np.load(f"{self.directory}/{hr_emg_path}")

        if self.augment:
            # Randomly choose whether to augment the data or not
            augmentation_method = random.choice(
                [self.no_augmentation, self.apply_gaussian_noise, self.apply_wavelet_decomposition, self.apply_magnitude_warping])

            # Apply the selected augmentation method
            hr_emg = augmentation_method(hr_emg)

        # Bandpass filter
        hr_emg = self.butter_bandpass_filter(hr_emg, 20, 500, 2048, order=2)

        # extract lr emg
        lr_emg = hr_emg[::2, ::2]
        lr_interpolated = self.grid_interpolation(lr_emg, hr_emg.shape, method='linear')

        # Standardize
        hr_emg = self.standardize(hr_emg)
        lr_emg = self.standardize(lr_emg)
        lr_interpolated = self.standardize(lr_interpolated)

        lr_emg = torch.tensor(lr_emg.copy()).to(torch.float32)
        hr_emg = torch.tensor(hr_emg.copy()).to(torch.float32)
        lr_interpolated = torch.tensor(lr_interpolated.copy()).to(torch.float32)

        # Convert input data to the same data type as the model parameters
        lr_emg = lr_emg.unsqueeze(0)
        hr_emg = hr_emg.unsqueeze(0)
        lr_interpolated = lr_interpolated.unsqueeze(0)

        return (lr_emg, hr_emg, hr_emg_path, lr_interpolated)

    def no_augmentation(self, data):
        return data

    def apply_gaussian_noise(self, data):
        return self.gaussian_noise_generator(data)

    def apply_wavelet_decomposition(self, data):
        return self.wavelet_decomposition(data)

    def apply_magnitude_warping(self,data):
        return self.magnitude_warping(data)

    def standardize(self, X):
        '''
        Standardize the input data
        Parameters
        ----------
        X: np.array
            Input data

        Returns
        -------
        X: np.array
            Standardized data
        '''
        return (X - self.mean) / self.std

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=2):
        """
        Apply a bandpass filter to the data
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
        y: np.array
            Filtered data
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, data)

        return filtered_signal

    def grid_interpolation(self, lr_emg: np.ndarray, hr_emg_shape: tuple, method: str) -> np.ndarray:
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

class CustomDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading
      and processing work in one place.
    '''

    def __init__(self, seq_len=1, batch_size=128, num_workers=2, data_directory: str = None, test_data_directory: str = None, augment: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_directory = data_directory
        self.test_data_directory = test_data_directory
        self.augment = augment
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.mean_std_dict = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # Load the JSON file with values for standardization into a Python dictionary
        with open(self.data_directory + '/mean_std.json', 'r') as file:
                self.mean_std_dict = json.load(file)

        data_train_val = pd.read_csv(self.data_directory + '/data.csv', sep=";")
        self.X_test = pd.read_csv(self.test_data_directory + '/data.csv', sep=";")

        self.X_train, self.X_val = train_test_split(data_train_val, train_size=0.8, test_size=0.2, shuffle = False)

    def train_dataloader(self):

        # Define your transformations
        data_transform = False

        train_dataset = TimeseriesDataset(self.X_train, self.data_directory, mean = self.mean_std_dict['mean'], std = self.mean_std_dict['std'], transform = data_transform, augment= self.augment)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers,
                                  drop_last=True,
                                  pin_memory=True
                                  )

        return train_loader

    def val_dataloader(self):

        # Define your transformations
        data_transform = False

        val_dataset = TimeseriesDataset(self.X_val,self.data_directory, mean = self.mean_std_dict['mean'], std = self.mean_std_dict['std'], transform = data_transform)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers,
                                drop_last=True,
                                pin_memory=True
                                )

        return val_loader

    def test_dataloader(self):

        # Define your transformations
        data_transform = False

        test_dataset = TimeseriesDataset(self.X_test,self.test_data_directory, mean = self.mean_std_dict['mean'], std = self.mean_std_dict['std'], transform = data_transform)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        return test_loader

