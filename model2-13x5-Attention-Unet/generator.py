"""
Author: Janina Kloos (janina.kloos@gmx.de)
Date created: 02.02.2024
File description: This file contains the class Generator, which is a convolutional neural network that is used to
generate high resolution data from low resolution data.

The code is called from the trainer.py file which is started by the train.py.
"""
import pywt
import ptwt
import torch.nn as nn
import torch
import numpy as np


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
    add the skip connection. Before the skip connection is added, a attention mechanism is applied.
    """

    def __init__(self, in_channels, out_channels, stride) -> None:
        super(UpConvolutionBlock, self).__init__()
        self.conv_input = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1,1),
                                         stride=(1,1,1))
        self.conv_skip = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1,1,1),
                                            stride=(1,1,2))

        self.relu = nn.ReLU()

        self.conv3 = nn.Conv3d(in_channels=out_channels, out_channels=1, kernel_size=(1,1,1),
                                            stride=(1,1,1))

        self.sigmoid = nn.Sigmoid()

        self.upsample_skip = nn.ConvTranspose3d(in_channels=1, out_channels=out_channels, kernel_size=(3, 3, 3),
                                          stride=stride, padding=(1, 1, 1),output_padding=(0, 0, 1))

        self.upsample_input = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                                                stride=stride, padding=(1, 1, 1), output_padding=(0, 0, 1))

        self.bn = nn.BatchNorm3d(out_channels)

        self.conv_last = nn.Sequential(
            nn.Conv3d(in_channels=out_channels + out_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.PReLU(),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.PReLU()
        )


    def forward(self, x: torch.tensor, skip: torch.tensor) -> torch.tensor:
        # go down in channels
        input = self.conv_input(x)

        # go down in time
        input_skip = self.conv_skip(skip)

        # add both
        attention = input+input_skip

        attention = self.relu(attention)

        # Go down in channels
        attention = self.conv3(attention)

        # Sigmoid for rearranging values
        attention = self.sigmoid(attention)

        # Upsample to skip connection shape
        attention = self.upsample_skip(attention)

        # elementwise multiplication with skip connection
        attention = attention*skip

        attention = self.bn(attention)

        # Upsample input
        x = self.upsample_input(x)

        out = torch.cat((x, attention), dim=1)

        out = self.conv_last(out)

        return out
class Generator(nn.Module):
    """
    This class is used to create the generator, which is a convolutional neural network that is used to generate high
    resolution data from low resolution data.
    """

    def __init__(self, num_res_blocks: int, num_res_blocks_last: int, wavelet, in_channels=1) -> None:
        super(Generator, self).__init__()

        # Convolution
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.PReLU()
            )

        # residual block 1
        self.res_blocks1 = nn.Sequential(*[ResidualBlock(in_channels=64) for _ in range(num_res_blocks)])

        # DownConvolution 1
        self.downConv1 = DownConvolutionBlock(in_channels=64, out_channels=128)

        # residual block 2
        self.res_blocks2 = nn.Sequential(*[ResidualBlock(in_channels=128) for _ in range(num_res_blocks)])

        # DownConvolution 2
        self.downConv2 = DownConvolutionBlock(in_channels=128, out_channels=256)

        # residual block 3
        self.res_blocks3 = nn.Sequential(*[ResidualBlock(in_channels=256) for _ in range(num_res_blocks)])

        # DownConvolution 3
        self.downConv3 = DownConvolutionBlock(in_channels=256, out_channels=512)

        # Upsampling of the residual and residual block 4

        self.res_blocks4 = nn.Sequential(*[ResidualBlock(in_channels=512) for _ in range(num_res_blocks)])

        # Upsampling 1
        self.upConv1 = UpConvolutionBlock(512, 256, stride=(1,1, 2))

        # Upsampling 2
        self.upConv2 = UpConvolutionBlock(256, 128, stride=(1, 1, 2))

        # Upsampling 3
        self.upConv3 = UpConvolutionBlock(128, 64, stride=(1, 1, 2))

        # Finale Convolution für Channel 1
        self.finalConv = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                   padding=(1, 1, 1))


    def forward(self, x: torch.tensor) -> torch.tensor:
        conv1 = self.conv1(x)
        conv1 = self.res_blocks1(conv1)
        skip1 = conv1
        #print("Generator output after conv1", conv1.shape)

        down1 = self.downConv1(conv1)
        down1 = self.res_blocks2(down1)
        #print("Generator output after DownConv 1", down1.shape)

        skip2 = down1

        down2 = self.downConv2(down1)
        down2 = self.res_blocks3(down2)
        skip3 = down2
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
