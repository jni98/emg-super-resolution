from torch.utils.data import Dataset
import torch
import numpy as np


class Custom_Dataset(Dataset):
    """
    Custom dataset class for loading the data

    """
    def __init__(self, data, mean, std):
        super().__init__()
        self.data = data

        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the data at the given index
        Parameters
        ----------
        index: int

        Returns
        -------
        lr_emg_standardized: torch.tensor
            Low resolution EMG data standardized
        hr_emg_standardized: torch.tensor
            High resolution EMG data standardized
        """

        if torch.is_tensor(index):
            index = index.tolist()

        lr_emg_path = self.data.iloc[index]['lr_filename']
        hr_emg_path = self.data.iloc[index]['hr_filename']

        lr_emg = np.load(self.data.iloc[index, 0])

        # Standardize using mean and std values of the complete dataset
        lr_emg_standardized = self.standardization(lr_emg, self.mean, self.std)
        # unsqueeze to add a one channel dimension at the beginning
        lr_emg_standardized = torch.tensor(lr_emg_standardized).unsqueeze(0).to(torch.float32)

        hr_emg = np.load(self.data.iloc[index, 1])

        #  Standardize using mean and std values of the complete dataset
        hr_emg_standardized = self.standardization(hr_emg,self.mean,self.std)
        # unsqueeze to add a one channel dimension at the beginning
        hr_emg_standardized = torch.tensor(hr_emg_standardized).unsqueeze(0).to(torch.float32)

        return lr_emg_standardized, hr_emg_standardized, lr_emg_path, hr_emg_path

    def standardization(self, data, mean, std):
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
        standardized_data = (data - mean)/std
        return standardized_data