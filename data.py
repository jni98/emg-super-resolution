from torch.utils.data import Dataset
import torch
import numpy as np


class Custom_Dataset(Dataset):

    def __init__(self, data, mode, min_value= np.inf, max_value = -np.inf):
        super().__init__()
        self.data = data
        self.min_value = min_value
        self.max_value = max_value

        if mode == 'train':
            # Load all data and calculate min/max values
            self._calculate_min_max()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        lr_emg_path = self.data.iloc[index]['lr_filename']
        hr_emg_path = self.data.iloc[index]['hr_filename']

        lr_emg = np.load(self.data.iloc[index, 0])
        # Normalize using min-max values of the complete dataset
        lr_emg_normalized = self.min_max_normalize(lr_emg, self.min_value, self.max_value)
        # unsqueeze to add a one channel dimension at the beginning
        lr_emg_normalized = torch.tensor(lr_emg_normalized).unsqueeze(0).to(torch.float32)

        hr_emg = np.load(self.data.iloc[index, 1])
        # Normalize using min-max values of the complete dataset
        hr_emg_normalized = self.min_max_normalize(hr_emg, self.min_value, self.max_value)
        # unsqueeze to add a one channel dimension at the beginning
        hr_emg_normalized = torch.tensor(hr_emg_normalized).unsqueeze(0).to(torch.float32)

        return lr_emg_normalized, hr_emg_normalized, lr_emg_path, hr_emg_path

    def _calculate_min_max(self):
        # Iterate through the dataset to calculate min and max values
        for index in range(len(self.data)):
            lr_emg_path = self.data.iloc[index]['lr_filename']
            hr_emg_path = self.data.iloc[index]['hr_filename']

            lr_emg = np.load(lr_emg_path)
            hr_emg = np.load(hr_emg_path)

            # Update min and max values
            self.min_value = min(self.min_value, lr_emg.min(), hr_emg.min())
            self.max_value = max(self.max_value, lr_emg.max(), hr_emg.max())

    def min_max_normalize(self, data, min_val, max_val, a=-1, b=1):
        # Min-max normalization using precalculated min and max values
        normalized_data = (b - a) * (data - min_val) / (max_val - min_val) + a
        return normalized_data
