import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # Elementwise sum
        return out


class Generator(nn.Module):
    def __init__(self, in_channels):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(9, 3), stride=(4, 1),
                               padding=(4, 1))  # padding changed for correct shapes
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

    def forward(self, x):
        print("input shape Generator", x.shape)
        out = self.conv1(x)
        out = self.prelu(out)

        residual = out

        out = self.res_blocks(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out += residual

        out = self.upsample(out)
        out = self.conv_output(out)

        print("output shape Generator", out.shape)

        return out
