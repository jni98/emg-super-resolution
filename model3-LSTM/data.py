import json

import torch
import numpy as np
import pytorch_lightning as pl
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


class TimeseriesDataset(Dataset):
    '''
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    '''
    def __init__(self, X: np.ndarray,directory: str, transform=None):
        self.X = X
        self.directory = directory
        self.transform = transform
    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> tuple:
        if torch.is_tensor(index):
            index = index.tolist()

        # Load file paths
        hr_emg_path = self.X.iloc[index]['hr_filename']

        # Load and preprocess HR data
        hr_emg = np.load(f"{self.directory}/{hr_emg_path}")

        # Bandpass filter
        hr_emg = self.butter_bandpass_filter(hr_emg, 20, 500, 2048, order=2)

        # extract lr emg
        lr_emg = hr_emg[::2, ::2]

        hr_emg = hr_emg.reshape(-1, hr_emg.shape[2])
        lr_emg = lr_emg.reshape(-1, lr_emg.shape[2])

        # Transform
        if self.transform:
            lr_emg = self.transform(lr_emg.copy()).squeeze()
            hr_emg = self.transform(hr_emg.copy()).squeeze()

        # Convert input data to the same data type as the model parameters
        lr_emg = lr_emg.to(torch.float32)
        hr_emg = hr_emg.to(torch.float32)

        return (lr_emg, hr_emg)

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

class CustomDataModule(pl.LightningDataModule):
    '''
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading
      and processing work in one place.
    '''

    def __init__(self, seq_len=1, batch_size=128, num_workers=0, data_directory: str = None, test_data_directory: str = None):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_directory = data_directory
        self.test_data_directory = test_data_directory
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
        data_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert data to tensor
            transforms.Normalize(mean=self.mean_std_dict['mean'], std=self.mean_std_dict['std'])  # Standardize data
        ])

        train_dataset = TimeseriesDataset(self.X_train, self.data_directory, transform = data_transform)
        train_loader = DataLoader(train_dataset,
                                  batch_size=self.batch_size,
                                  shuffle=False,
                                  num_workers=self.num_workers)

        return train_loader

    def val_dataloader(self):

        # Define your transformations
        data_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert data to tensor
            transforms.Normalize(mean=self.mean_std_dict['mean'], std=self.mean_std_dict['std'])  # Standardize data
        ])

        val_dataset = TimeseriesDataset(self.X_val,self.data_directory, transform = data_transform)
        val_loader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.num_workers)

        return val_loader

    def test_dataloader(self):

        # Define your transformations
        data_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert data to tensor
            transforms.Normalize(mean=self.mean_std_dict['mean'], std=self.mean_std_dict['std'])  # Standardize data
        ])

        test_dataset = TimeseriesDataset(self.X_test,self.test_data_directory, transform = data_transform)
        test_loader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=self.num_workers)

        return test_loader

