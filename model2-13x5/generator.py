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
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(in_channels)

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

        # input: 1x7x3x2048
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.prelu1 = nn.PReLU()

        # input: 64x7x3x2048
        self.resUp1 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1))

        self.res_blocks1 = nn.Sequential(*[ResidualBlock(64) for _ in range(5)])

        # input: 64x7x3x2048
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1,2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(128)
        self.prelu2 = nn.PReLU()

        self.resUp2 = nn.ConvTranspose3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1))

        self.res_blocks2 = nn.Sequential(*[ResidualBlock(128) for _ in range(5)])

        # input: 128x7x3x1024
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(256)
        self.prelu3 = nn.PReLU()

        self.resUp3 = nn.ConvTranspose3d(256, 256, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1))

        self.res_blocks3 = nn.Sequential(*[ResidualBlock(256) for _ in range(5)])

        # input: 256x7x3x512
        self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1))
        self.bn4 = nn.BatchNorm3d(512)
        self.prelu4 = nn.PReLU()

        self.resUp4 = nn.ConvTranspose3d(512, 512, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1))

        self.res_blocks4 = nn.Sequential(*[ResidualBlock(512) for _ in range(5)])

        # input: 512x7x3x256
        self.upsample1 = nn.ConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(2, 2,2), padding=(1, 1, 1), output_padding=(0, 0, 1))
        self.bn5 = nn.BatchNorm3d(256)
        self.prelu5 = nn.PReLU()

        # input: 256x13x5x512
        self.upsample2 = nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1), output_padding=(0, 0, 1))
        self.bn6 = nn.BatchNorm3d(128)
        self.prelu6 = nn.PReLU()

        # input: 128x13x5x1024
        self.upsample3 = nn.ConvTranspose3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 1, 2), padding=(1, 1, 1), output_padding=(0, 0, 1))
        self.bn7 = nn.BatchNorm3d(64)
        self.prelu7 = nn.PReLU()

        # input: 64x13x5x2048
        self.conv5 = nn.Conv3d(64, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

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

        down1 = self.conv1(x)
        down1 = self.bn1(down1)
        down1 = self.prelu1(down1)
        print("Generator output after conv1", down1.shape)

        res1 = self.resUp1(down1)
        print("Generator output after resUp1", res1.shape)

        down1 = self.res_blocks1(down1)

        down2 = self.conv2(down1)
        down2 = self.bn2(down2)
        down2 = self.prelu2(down2)
        print("Generator output after conv2", down2.shape)

        res2 = self.resUp2(down2)
        print("Generator output after resUp2", res2.shape)

        down2 = self.res_blocks2(down2)

        down3 = self.conv3(down2)
        down3 = self.bn3(down3)
        down3 = self.prelu3(down3)
        print("Generator output after conv3", down3.shape)

        res3= self.resUp3(down3)
        print("Generator output after resUp3", res3.shape)

        down3 = self.res_blocks3(down3)

        down4 = self.conv4(down3)
        down4 = self.bn4(down4)
        down4 = self.prelu4(down4)
        print("Generator output after conv4", down4.shape)

        res4 = self.resUp4(down4)
        print("Generator output after resUp4", res4.shape)

        up1 = self.upsample1(down4)
        up1 = self.bn5(up1)
        up1 = self.prelu5(up1)
        print("Generator output after upsample1", up1.shape)

        up2 = self.upsample2(up1)
        up2 = self.bn6(up2)
        up2 = self.prelu6(up2)
        print("Generator output after upsample2", up2.shape)

        up3 = self.upsample3(up2)
        up3 = self.bn7(up3)
        up3 = self.prelu7(up3)
        print("Generator output after upsample3", up3.shape)

        out = self.conv5(up3)

        print("Generator output", out.shape)

        return out