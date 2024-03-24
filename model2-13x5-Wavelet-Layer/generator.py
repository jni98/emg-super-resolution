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

    def __init__(self, wavelet) -> None:
        super(DownConvolutionBlock, self).__init__()
        self.wavelet = wavelet

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Calulate the wavelet transform coefficients
        coeffs = ptwt.wavedec(x, self.wavelet, mode='zero', level=1)

        # Concatenate the coefficients in channel dimension
        out = torch.cat((coeffs[0], coeffs[1]), dim=1)

        return coeffs, out


class UpConvolutionBlock(nn.Module):
    """
    This class is used to create an up convolution block, which is used in the generator to upsample the data and also
    add the skip connection.
    """

    def __init__(self, wavelet) -> None:
        super(UpConvolutionBlock, self).__init__()
        self.wavelet = wavelet

    def forward(self, x: torch.tensor, skip: torch.tensor) -> torch.tensor:
        # Calculate the midpoint along the second dimension
        midpoint = x.shape[1] // 2

        # Split the tensor into two parts at the midpoint
        coeff1 = x[:, :midpoint]
        coeff2 = x[:, midpoint:]

        # Add the tensors to a list
        coeff_list = [coeff1, coeff2]

        # upsample
        up = ptwt.waverec(coeff_list, self.wavelet)

        # sum up with skip connection
        out = up + skip

        # TODO: Add convolutions here
        # convolve to get channel number as before the torch.cat
        # out = self.conv(out)
        # out = self.bn2(out)
        # out = self.prelu2(out)

        return out


class Generator(nn.Module):
    """
    This class is used to create the generator, which is a convolutional neural network that is used to generate high
    resolution data from low resolution data.
    """

    def __init__(self, num_res_blocks: int, num_res_blocks_last: int, wavelet, in_channels=1) -> None:
        super(Generator, self).__init__()

        # Define wavelet for dwt
        self.wavelet = wavelet

        # Convolution
        self.conv1 = nn.Conv3d(in_channels, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.prelu1 = nn.PReLU()

        # Upsampling of the residual and residual block 1
        #TODO: Change convolution so that skip connection fits result from inverse wavelet
        self.residualUp1 = nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 2), stride=(2, 2, 1),
                                              padding=(1, 1, 0))
        self.res_blocks1 = nn.Sequential(*[ResidualBlock(in_channels=64) for _ in range(num_res_blocks)])

        # DownConvolution 1
        self.downConv1 = DownConvolutionBlock(wavelet=self.wavelet)

        # Upsampling of the residual and residual block 2
        self.residualUp2 = nn.ConvTranspose3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3),
                                              stride=(2, 2, 1), padding=(1, 1, 0))
        self.res_blocks2 = nn.Sequential(*[ResidualBlock(in_channels=128) for _ in range(num_res_blocks)])

        # DownConvolution 2
        self.downConv2 = DownConvolutionBlock(wavelet=self.wavelet)

        # Upsampling of the residual and residual block 3
        self.residualUp3 = nn.ConvTranspose3d(256, 256, kernel_size=(3, 3, 2), stride=(2, 2, 1), padding=(1, 1, 0))
        self.res_blocks3 = nn.Sequential(*[ResidualBlock(in_channels=256) for _ in range(num_res_blocks)])

        # DownConvolution 3
        self.downConv3 = DownConvolutionBlock(wavelet=self.wavelet)

        # Upsampling of the residual and residual block 4

        self.res_blocks4 = nn.Sequential(*[ResidualBlock(in_channels=512) for _ in range(num_res_blocks_last)])

        # Upconv of spatial dimension
        self.upConv = nn.ConvTranspose3d(in_channels=512, out_channels=512, kernel_size=(3,3,3),
                                         stride=(2,2,1), padding=(1,1,1))
        self.bn2 = nn.BatchNorm3d(512)
        self.prelu2 = nn.PReLU()

        # Upsampling 1
        self.upConv1 = UpConvolutionBlock(wavelet=self.wavelet)

        # Upsampling 2
        self.upConv2 = UpConvolutionBlock(wavelet=self.wavelet)

        # Upsampling 3
        self.upConv3 = UpConvolutionBlock(wavelet=self.wavelet)

        # Finale Convolution für Channel 1
        self.finalConv = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                   padding=(1, 1, 1))

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = torch.randn(torch.Size([1, 1, 7, 3, 2048])).to('cuda')
        conv1 = self.conv1(x)
        conv1 = self.bn1(conv1)
        conv1 = self.prelu1(conv1)
        conv1 = self.res_blocks1(conv1)
        skip1 = self.residualUp1(conv1)
        # print("Generator output after conv1", conv1.shape)

        coeffs1, down1 = self.downConv1(conv1)
        down1 = self.res_blocks2(down1)
        skip2 = self.residualUp2(down1)
        # print("Generator output after DownConv 1", down1.shape)

        coeffs2, down2 = self.downConv2(down1)
        down2 = self.res_blocks3(down2)
        skip3 = self.residualUp3(down2)
        # print("Generator output after DownConv 2", down2.shape)

        coeffs3, down3 = self.downConv3(down2)
        down3 = self.res_blocks4(down3)
        # print("Generator output after DownConv 3", down3.shape)

        up = self.upConv(down3)
        up = self.bn2(up)
        up = self.prelu2(up)

        up1 = self.upConv1(up, skip3)
        # print("Generator output after upsample1", up1.shape)
        up2 = self.upConv2(up1, skip2)
        # print("Generator output after upsample2", up2.shape)
        up3 = self.upConv3(up2, skip1)
        # print("Generator output after upsample3", up3.shape)

        out = self.finalConv(up3)

        # print("Generator output", out.shape)

        return out
