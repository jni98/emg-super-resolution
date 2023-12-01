from torch.utils.data import Dataset
import torch
import numpy as np


class Dataset(Dataset):

    def __init__(self, data, mode):
        super().__init__()
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        lr_emg = np.load(self.data.iloc[index, 0])

        # unsqueeze to add a one channel dimension at the beginning
        lr_emg = torch.tensor(lr_emg).unsqueeze(0).to(torch.float32)

        hr_emg = np.load(self.data.iloc[index, 1])

        # unsqueeze to add a one channel dimension at the beginning
        hr_emg = torch.tensor(hr_emg).unsqueeze(0).to(torch.float32)

        return (lr_emg, hr_emg)
