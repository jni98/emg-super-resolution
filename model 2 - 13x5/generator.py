"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Generator, which is a convolutional neural network that is used to
generate high resolution data from low resolution data.

The code is called from the trainer.py file which is started by the train.py.
"""
import torch.nn as nn
import torch
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        """
         Constructs all the necessary attributes for the residual block object.
        Parameters
        ----------
        in_channels: int
                     Number of channels of the input tensor.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forwards the input tensor through all the layers.

        Parameters
        ----------
        x: torch.tensor
           Tensor that is given to the network.

        Returns
        -------
        out: torch.tensor
             Output tensor of the network.
        """
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        return out


class Generator(nn.Module):
    def __init__(self, in_channels=1) -> None:
        """
        Constructs all the necessary attributes for the generator object.
        Parameters
        ----------
        in_channels: int
                     Number of channels the emg data has. On default set to 1.
        """
        super(Generator, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))  # Adjust padding
        self.prelu = nn.PReLU()

        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(17)])

        self.conv2 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1,2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1))

        self.conv4 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1))

        # Upsampling layers using transposed convolutions
        self.upsample1 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2,2), padding=(1, 1, 1), output_padding=(0, 0, 1))

        # Upsample last dimension to go from 817 to 2048
        self.upsample2 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1), output_padding=(0, 0, 1))

        self.upsample3 = nn.ConvTranspose3d(64, 1, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1), output_padding=(0, 0, 1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forwards the input tensor through all the layers.

        Parameters
        ----------
        x: torch.tensor
            Tensor that is given to the network.

        Returns
        -------
        out: torch.tensor
            Output tensor of the network.
        """
        print("Generator input", x.shape)

        out = self.conv1(x)
        out = self.prelu(out)
        print("Generator after conv1", out.shape)

        residual = out
        print("Generator residual", residual.shape)

        out = self.res_blocks(out)
        print("Generator after res_blocks", out.shape)

        out += residual

        out = self.conv2(out)
        out = self.bn2(out)
        print("Generator after conv2", out.shape)

        out = self.conv3(out)
        print("Generator after conv3", out.shape)

        out = self.conv4(out)
        print("Generator after conv4", out.shape)

        # Upsampling
        out = self.upsample1(out)
        print("Generator after upsample1", out.shape)

        out = self.upsample2(out)
        print("Generator after upsample2", out.shape)

        out = self.upsample3(out)
        print("Generator after upsample3", out.shape)

        return out