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
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

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
        out += residual  # Elementwise sum
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
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(9, 3), stride=(4, 1), padding=(4, 1))
        self.prelu = nn.PReLU()

        # Adding residual blocks
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(16)])

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Upsampling layers
        # PixelShuffling: rearrange the tensor of shape (N,C,H,W) into (N,C/r*r,H*r,W*r) where r is the shuffling factor.
        # It basically convert the depth(channel) into the space(height and width). In Generator, pixelshuffling is used
        # to upsample the images size.
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # Upsampling factor 4
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),  # Upsampling factor 4
            nn.PReLU()
        )

        self.conv_output = nn.Conv2d(64, 1, kernel_size=(9, 3), stride=1, padding=(4, 1))

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

        # reshape to 2D
        reshaped_input = x.reshape(x.shape[0], x.shape[1],x.shape[2] * x.shape[3], x.shape[4])

        # transpose to have samples x channels
        reshaped_input = reshaped_input.transpose(-1,-2)
        print("Reshaped input shape", reshaped_input.shape)

        out = self.conv1(reshaped_input)
        print("Generator conv1 output", out.shape)

        out = self.prelu(out)

        residual = out

        out = self.res_blocks(out)
        print("Generator res_blocks output", out.shape)

        out = self.conv2(out)
        print("Generator conv2 output", out.shape)

        out = self.bn2(out)
        print("Generator bn2 output", out.shape)

        out += residual
        print("Generator out += residual output", out.shape)

        out = self.upsample(out)
        print("Generator upsample output", out.shape)

        out = self.conv_output(out)
        print("Generator conv_output output", out.shape)

        # transpose for channels x samples
        reshaped_output = out.transpose(-1,-2)

        # reshape to have 14x6 instead of 84
        reshaped_output = reshaped_output.reshape(reshaped_output.shape[0], reshaped_output.shape[1], x.shape[2] * 2, x.shape[3] *2, reshaped_output.shape[-1])

        # delete last row and column
        reshaped_output = reshaped_output[:,:,:-1,:-1, :]

        return reshaped_output
