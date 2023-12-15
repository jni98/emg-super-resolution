from torch.utils.data import Dataset
import torch
import numpy as np


class Custom_Dataset(Dataset):

    def __init__(self, data, mode):
        super().__init__()
        self.data = data
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        lr_emg_path = self.data.iloc[index]['lr_filename']
        hr_emg_path = self.data.iloc[index]['hr_filename']


        try:
            lr_emg = np.load(self.data.iloc[index, 0])
            # Continue with your code using lr_emg
        except Exception as e:
            print(self.data.iloc[index, 0])

        # unsqueeze to add a one channel dimension at the beginning
        lr_emg = torch.tensor(lr_emg).unsqueeze(0).to(torch.float32)

        try:
            hr_emg = np.load(self.data.iloc[index, 1])
        except Exception as e:
            print(self.data.iloc[index, 1])

        # unsqueeze to add a one channel dimension at the beginning
        hr_emg = torch.tensor(hr_emg).unsqueeze(0).to(torch.float32)

        return lr_emg, hr_emg, lr_emg_path, hr_emg_path