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
        """
        Constructs all the necessary attributes for the residual block object.
        Parameters
        ----------
        in_channels: int
                        Number of channels of the input tensor.
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                      padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(in_channels)


    def forward(self, x: torch.tensor) -> torch.tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        return out


class DownConvolutionBlock(nn.Module):
    """
    This class is used to create a down convolution block, which is used in the generator to downsample the data.
    """

    def __init__(self, in_channels, out_channels) -> None:
        """
        Constructs all the necessary attributes for the down convolution block object.
        Parameters
        ----------
        in_channels: int
                        Number of channels of the input tensor.
        out_channels: int
                        Number of channels of the output tensor.
        """
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
        """
        Constructs all the necessary attributes for the up convolution block object.
        Parameters
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        stride: tuple
            Stride for the transposed convolution
        """
        super(UpConvolutionBlock, self).__init__()
        self.upConv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                                         stride=stride, padding=(1, 1, 1), output_padding=(0, 0, 1)),
            nn.BatchNorm3d(out_channels),
            nn.PReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=out_channels + out_channels, out_channels=out_channels, kernel_size=(3, 3, 3),
                      stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(out_channels),
            nn.PReLU()
        )



    def forward(self, x: torch.tensor, skip: torch.tensor) -> torch.tensor:
        # upsample
        out = self.upConv(x)

        # concatenate with skip connection
        out = torch.cat((out, skip), dim=1)

        # convolve to get channel number as before the torch.cat
        out = self.conv(out)

        return out

class Generator(nn.Module):
    """
    This class is used to create the generator, which is a convolutional neural network that is used to generate high
    resolution data from low resolution data.
    """

    def __init__(self, num_res_blocks: int, in_channels=1) -> None:
        """
        Constructs all the necessary attributes for the generator object.
        Parameters
        ----------
        num_res_blocks: int
            Number of residual blocks in the generator.
        in_channels:
            Number of input channels.
        """
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

        # Just a normal convolution, since changing of channels is not necessary when only taking real part
        self.channel_down = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.PReLU()
        )

        # Upsampling 3
        self.upConv3 = UpConvolutionBlock(128, 64, stride=(1, 1, 2))

        # Final convolution
        self.finalConv = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                                   padding=(1, 1, 1))

    def fft(self, x: torch.tensor) -> torch.tensor:
        """
        This function is used to perform a Fast Fourier Transform on the input tensor.
        Parameters
        ----------
        x: torch.tensor
            Input tensor to perform the FFT on.

        Returns
        -------
        real_part: torch.tensor
            Real part of the FFT
        """
        # Calculate FFT of the signal
        fft_signal = torch.fft.fft(x, dim=-1)

        # Extract real and imaginary parts
        real_part = torch.real(fft_signal)

        # Only use real part of the signal
        return real_part

    def ifft(self, x: torch.tensor) -> torch.tensor:
        """
        This function is used to perform an Inverse Fast Fourier Transform on the input tensor.
        Parameters
        ----------
        x: torch.tensor
            Input tensor to perform the iFFT on.

        Returns
        -------
        real_part: torch.tensor
            Real part of the iFFT
        """

        # Perform inverse FFT
        ifft_signal = torch.fft.ifftn(x, dim=-1)
        print("Generator output after ifft", ifft_signal.shape)

        # Extract real and imaginary parts
        real_part = torch.real(ifft_signal)

        # Only return real part
        return real_part

    def forward(self, x: torch.tensor) -> torch.tensor:
        conv1 = self.conv1(x)
        conv1 = self.res_blocks1(conv1)
        skip1 = conv1
        #print("Generator output after conv1", conv1.shape)

        down1 = self.downConv1(conv1)
        down1 = self.res_blocks2(down1)
        #print("Generator output after DownConv 1", down1.shape)

        down1_fft= self.fft(down1)
        #print("Generator output after FFT", down1_fft.shape)

        skip2 = down1_fft

        down2 = self.downConv2(down1_fft)
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

        up2_ifft = self.ifft(up2)

        #channel_down = self.channel_down(up2_ifft)

        up3 = self.upConv3(up2_ifft, skip1)
        #print("Generator output after upsample3", up3.shape)

        out = self.finalConv(up3)

        #print("Generator output", out.shape)

        return out