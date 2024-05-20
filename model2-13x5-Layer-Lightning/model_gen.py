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
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    This class is used to create a residual block, which is used in the generator.
    """

    def __init__(self, in_channels) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(in_channels),
            nn.PReLU())

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1)),
            nn.BatchNorm3d(in_channels))

        self.prelu = nn.PReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual
        out = self.prelu(out)
        return out


class DownConvolutionBlock(nn.Module):
    """
    This class is used to create a down convolution block, which is used in the generator to downsample the data.
    """

    def __init__(self, in_channels, out_channels) -> None:
        super(DownConvolutionBlock, self).__init__()
        self.downConv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                                  stride=(1, 1, 2), padding=(1, 1, 1))
        self.bn = nn.BatchNorm3d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.downConv(x)
        out = self.bn(out)
        out = self.prelu(out)
        return out


class UpConvolutionBlock(nn.Module):
    """
    This class is used to create an up convolution block, which is used in the generator to upsample the data and also
    add the skip connection.
    """
    def __init__(self, in_channels, out_channels, stride) -> None:
        super(UpConvolutionBlock, self).__init__()
        self.upConv = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                                         stride=stride, padding=(1, 1, 1), output_padding=(0, 0, 1))
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.prelu1 = nn.PReLU()

        self.conv = nn.Conv3d(in_channels=out_channels + out_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                              stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.prelu2 = nn.PReLU()

    def forward(self, x: torch.tensor, skip: torch.tensor) -> torch.tensor:
        # upsample
        out = self.upConv(x)
        out = self.bn1(out)
        out = self.prelu1(out)

        # concatenate with skip connection
        out = torch.cat((out, skip), dim=1)

        # convolve to get channel number as before the torch.cat
        out = self.conv(out)
        out = self.bn2(out)
        out = self.prelu2(out)

        return out


class Generator(nn.Module):
    """
    This class is used to create the generator, which is a convolutional neural network that is used to generate high
    resolution data from low resolution data.
    """

    def __init__(self,num_res_blocks: int, num_res_blocks_last:int, in_channels=1) -> None:

        super(Generator, self).__init__()

        # Convolution
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.prelu1 = nn.PReLU()

        # Upsampling of the residual and residual block 1
        self.residualUp1 = nn.ConvTranspose3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1))
        self.res_blocks1 = nn.Sequential(*[ResidualBlock(in_channels=64) for _ in range(num_res_blocks)])

        # DownConvolution 1
        self.downConv1 = DownConvolutionBlock(in_channels=64, out_channels=128)

        # Upsampling of the residual and residual block 2
        self.residualUp2 = nn.ConvTranspose3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1))
        self.res_blocks2 = nn.Sequential(*[ResidualBlock(in_channels=128) for _ in range(num_res_blocks)])

        # DownConvolution 2
        self.downConv2 = DownConvolutionBlock(in_channels=128, out_channels=256)

        # Upsampling of the residual and residual block 3
        self.residualUp3 = nn.ConvTranspose3d(256, 256, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1))
        self.res_blocks3 = nn.Sequential(*[ResidualBlock(in_channels=256) for _ in range(num_res_blocks)])

        # DownConvolution 3
        self.downConv3 = DownConvolutionBlock(in_channels=256, out_channels=512)

        # Upsampling of the residual and residual block 4
        self.residualUp4 = nn.ConvTranspose3d(512, 512, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(1, 1, 1))
        self.res_blocks4 = nn.Sequential(*[ResidualBlock(in_channels=512) for _ in range(num_res_blocks_last)])

        # Upsampling 1
        self.upConv1 = UpConvolutionBlock(512, 256, stride=(2, 2, 2))

        # Upsampling 2
        self.upConv2 = UpConvolutionBlock(256, 128, stride=(1, 1, 2))

        # Upsampling 3
        self.upConv3 = UpConvolutionBlock(128, 64, stride=(1, 1, 2))

        # Finale Convolution für Channel 1
        self.finalConv = nn.Conv3d(64, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x: torch.tensor) -> torch.tensor:

        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.prelu1(conv1)
        conv1 = self.res_blocks1(conv1)
        skip1 = self.residualUp1(conv1)
        #print("Generator output after conv1", conv1.shape)

        down1 = self.downConv1(conv1)
        down1 = self.res_blocks2(down1)
        skip2 = self.residualUp2(down1)
        #print("Generator output after DownConv 1", down1.shape)

        down2 = self.downConv2(down1)
        down2 = self.res_blocks3(down2)
        skip3 = self.residualUp3(down2)
        #print("Generator output after DownConv 2", down2.shape)

        down3 = self.downConv3(down2)
        down3 = self.res_blocks4(down3)
        #print("Generator output after DownConv 3", down3.shape)

        up1 = self.upConv1(down3, skip3)
        #print("Generator output after upsample1", up1.shape)
        up2 = self.upConv2(up1, skip2)
        #print("Generator output after upsample2", up2.shape)
        up3 = self.upConv3(up2, skip1)
        #print("Generator output after upsample3", up3.shape)

        out = self.finalConv(up3)

        #print("Generator output", out.shape)

        return out