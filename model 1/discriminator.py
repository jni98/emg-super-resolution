import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.leaky_relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.leaky_relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(128)
        self.leaky_relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.leaky_relu5 = nn.LeakyReLU(0.2, inplace=True)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(256)
        self.leaky_relu6 = nn.LeakyReLU(0.2, inplace=True)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.batch_norm7 = nn.BatchNorm2d(512)
        self.leaky_relu7 = nn.LeakyReLU(0.2, inplace=True)

        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.batch_norm8 = nn.BatchNorm2d(512)
        self.leaky_relu8 = nn.LeakyReLU(0.2, inplace=True)

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(512 * 32 * 8, 1024)
        self.leaky_relu9 = nn.LeakyReLU(0.2, inplace=True)
        self.dense2 = nn.Linear(1024, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.leaky_relu1(out)

        out = self.conv2(out)
        out = self.batch_norm2(out)
        out = self.leaky_relu2(out)

        out = self.conv3(out)
        out = self.batch_norm3(out)
        out = self.leaky_relu3(out)

        out = self.conv4(out)
        out = self.batch_norm4(out)
        out = self.leaky_relu4(out)

        out = self.conv5(out)
        out = self.batch_norm5(out)
        out = self.leaky_relu5(out)

        out = self.conv6(out)
        out = self.batch_norm6(out)
        out = self.leaky_relu6(out)

        out = self.conv7(out)
        out = self.batch_norm7(out)
        out = self.leaky_relu7(out)

        out = self.conv8(out)
        out = self.batch_norm8(out)
        out = self.leaky_relu8(out)

        out = self.flatten(out)

        out = self.dense1(out)
        out = self.leaky_relu9(out)
        out = self.dense2(out)
        out = self.activation(out)

        return out
