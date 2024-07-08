import torch
from torch.utils.data import Dataset
import numpy as np
from emg_augmentations import GaussianNoise, WaveletDecomposition
import random
from utils import butter_bandpass_filter, grid_interpolation, standardization


class Custom_Dataset(Dataset):
    """
    Custom dataset class for loading the data
    """

    def __init__(self, data, directory, mean: float, std: float, augment: bool, bandpass_upper:int) -> None:
        """
        Constructs all the necessary attributes for the custom dataset object.
        Parameters
        ----------
        data: pandas.DataFrame
                Dataframe containing the file paths of the data.
        directory: str
                Directory where the data is stored.
        mean: float
                Mean of the data.
        std: float
                Standard deviation of the data.
        augment: bool
                Whether to augment the data or not.
        bandpass_upper: int
                Upper bandpass frequency.
        """
        super().__init__()
        self.data = data
        self.directory = directory
        self.mean = mean
        self.std = std
        self.augment = augment
        self.bandpass_upper = bandpass_upper

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
        hr_emg = butter_bandpass_filter(hr_emg, 20, self.bandpass_upper, 2048, order=2)

        # extract lr emg
        lr_emg = hr_emg[::2, ::2]

        # interpolate for hr shape
        lr_emg = grid_interpolation(lr_emg, hr_emg.shape, method='linear')

        # Create lr file path
        lr_emg_path = hr_emg_path.replace("hr", "lr")

        # standardize both
        lr_emg_standardized = standardization(lr_emg, self.mean, self.std)
        hr_emg_standardized = standardization(hr_emg, self.mean, self.std)

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