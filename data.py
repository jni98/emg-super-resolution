from torch.utils.data import Dataset
import torch
import numpy as np


class Custom_Dataset(Dataset):

    def __init__(self, data, min_value, max_value):
        super().__init__()
        self.data = data
        self.min_value = min_value
        self.max_value = max_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

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

    def min_max_normalize(self, data, min_val, max_val, a=-1, b=1):
        # Min-max normalization using precalculated min and max values
        normalized_data = (b - a) * (data - min_val) / (max_val - min_val) + a
        return normalized_data
