import torch
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import butter, filtfilt
from torch.utils.data import Dataset
import numpy as np
from emg_augmentations import GaussianNoise, WaveletDecomposition
import random


class Custom_Dataset(Dataset):
    """
    Custom dataset class for loading the data
    """

    def __init__(self, data, directory, mean: float, std: float, augment: bool) -> None:
        super().__init__()
        self.data = data
        self.directory = directory
        self.mean = mean
        self.std = std
        self.augment = augment

        self.gaussian_noise_generator = GaussianNoise(target_snr__db=5.0)
        self.wavelet_decomposition = WaveletDecomposition(b=0.25, wavelet="db7", level=5, nr_of_grids=1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        if torch.is_tensor(index):
            index = index.tolist()

        # Load file paths
        hr_emg_path = self.data.iloc[index]['hr_filename']

        # Load and preprocess HR data
        hr_emg = np.load(f"{self.directory}/{hr_emg_path}")

        if self.augment:
            # Randomly choose whether to augment the data or not
            augmentation_method = random.choice(
                [self.no_augmentation, self.apply_gaussian_noise, self.apply_wavelet_decomposition])

            # Apply the selected augmentation method
            hr_emg = augmentation_method(hr_emg)

        # Bandpass filter
        hr_emg = self.butter_bandpass_filter(hr_emg, 20, 400, 2048, order=2)

        # extract lr emg
        lr_emg = hr_emg[::2, ::2]

        # Create lr file path
        lr_emg_path = hr_emg_path.replace("hr", "lr")

        # standardize both
        lr_emg_standardized = self.standardization(lr_emg, self.mean, self.std)
        hr_emg_standardized = self.standardization(hr_emg, self.mean, self.std)

        # Change to torch and shape with channel = 1
        lr_emg_standardized_tensor = torch.tensor(lr_emg_standardized).unsqueeze(0).to(torch.float32)
        hr_emg_standardized_tensor = torch.tensor(hr_emg_standardized).unsqueeze(0).to(torch.float32)

        return lr_emg_standardized_tensor, hr_emg_standardized_tensor, lr_emg_path, hr_emg_path

    def no_augmentation(self, data):
        return data

    def apply_gaussian_noise(self, data):
        return self.gaussian_noise_generator(data)

    def apply_wavelet_decomposition(self, data):
        return self.wavelet_decomposition(data)

    def standardization(self, data: np.array, mean: float, std: float) -> np.array:
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

