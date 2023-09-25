import torch
from torch import nn


class DummyNetwork(nn.Module):
    # This is a dummy network, such that the code runs
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(512 * 512, 256)
        self.layer_2 = nn.Linear(256, 16)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.float()
        batch_size, channels, width, height = x.size()
        x = x[:, :, :, 0]  # Only select the first slice
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        X = self.relu(x)
        x = self.layer_2(x)
        return x


class ConvDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bottleneck=False):
        super(ConvDownBlock, self).__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
        )

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
        )

        self.norm1 = nn.BatchNorm3d(out_channels)
        self.norm2 = nn.BatchNorm3d(out_channels)

        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, x):
        y = self.norm1(self.conv1(x)).relu()
        y = self.norm2(self.conv2(y)).relu()
        return self.maxpool(y)


class ConvUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUpBlock, self).__init__()

        self.deconv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2, 2, 2),
            stride=2,
        )

        self.conv1 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
        )

        self.norm1 = nn.BatchNorm3d(num_features=out_channels)
        self.norm2 = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        y = self.deconv(x)
        y = self.norm1(self.conv1(y)).relu()
        return self.norm2(self.conv2(y).relu())


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super(UNet3D, self).__init__()

        self.down_block1 = ConvDownBlock(in_channels, 32)
        self.down_block2 = ConvDownBlock(32, 64)
        self.down_block3 = ConvDownBlock(64, 128)
        self.down_block4 = ConvDownBlock(128, 256)

        self.up_block4 = ConvUpBlock(256, 128)
        self.up_block3 = ConvUpBlock(128, 64)
        self.up_block2 = ConvUpBlock(64, 32)
        self.up_block1 = ConvUpBlock(32, out_channels)

    def forward(self, x):
        y = self.down_block1(x)
        y = self.down_block2(y)
        y = self.down_block3(y)
        y = self.down_block4(y)

        y = self.up_block4(y)
        y = self.up_block3(y)
        y = self.up_block2(y)
        return self.up_block1(y)
