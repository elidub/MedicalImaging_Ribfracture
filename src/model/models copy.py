import torch
import torch.nn as nn


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
    def __init__(self, in_channels, out_channels):
        super().__init__()

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
        super().__init__()

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
        super().__init__()

        self.down_block1 = ConvDownBlock(in_channels, 32)
        self.down_block2 = ConvDownBlock(32, 64)
        self.down_block3 = ConvDownBlock(64, 128)
        self.down_block4 = ConvDownBlock(128, 256)

        self.up_block4 = ConvUpBlock(256, 128)
        self.up_block3 = ConvUpBlock(128, 64)
        self.up_block2 = ConvUpBlock(64, 32)
        self.up_block1 = ConvUpBlock(32, out_channels)

    def forward(self, x):
        x = x.unsqueeze(1)
        d1 = self.down_block1(x)
        d2 = self.down_block2(d1)
        d3 = self.down_block3(d2)
        d4 = self.down_block4(d3)

        y = self.up_block4(d4) + d3
        y = self.up_block3(y) + d2
        y = self.up_block2(y) + d1
        y = self.up_block1(y)
        y = y.squeeze(1)
        return torch.sigmoid(y)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=1,
            bias=False,
        )

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3, 3),
            padding=1,
            bias=False,
        )

        self.norm1 = nn.BatchNorm3d(out_channels)
        self.norm2 = nn.BatchNorm3d(out_channels)

        self.downsample = None

        if stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1, 1),
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        residual = x
        y = self.norm1(self.conv1(x)).relu()
        y = self.norm2(self.conv2(y)).relu()

        if self.downsample != None:
            residual = self.downsample(x)

        return (y + residual).relu()


class ResNet3D(nn.Module):
    def __init__(self, in_channels, configuration="resnet18"):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, 64, kernel_size=7, stride=1, padding=3)
        self.norm = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        if configuration == "resnet18":
            layers = [2, 2, 2, 2]
        elif configuration == "resnet34":
            layers = [3, 4, 6, 3]

        self.layer1 = self._build_layer(64, 64, num_blocks=layers[0], stride=1)
        self.layer2 = self._build_layer(64, 128, num_blocks=layers[1], stride=2)
        self.layer3 = self._build_layer(128, 256, num_blocks=layers[2], stride=2)
        self.layer4 = self._build_layer(256, 512, num_blocks=layers[3], stride=2)

    def _build_layer(self, in_channels, out_channels, num_blocks=2, stride=1):
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]

        for i in range(1, num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*blocks)

    def forward(self, x):
        y0 = self.maxpool(self.norm(self.conv(x)).relu())
        y1 = self.layer1(y0)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)

        return y2, y3, y4


class PyramidFeatures3D(nn.Module):
    def __init__(self, c3_size, c4_size, c5_size, feature_size=256):
        super().__init__()

        # Upsample C5 to get P5
        self.p5_conv1 = nn.Conv3d(c5_size, feature_size, 1, stride=1, padding=0)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p5_conv2 = nn.Conv3d(feature_size, feature_size, 3, stride=1, padding=1)

        # Upsample C4 and add P5 to get P4
        self.p4_conv1 = nn.Conv3d(c4_size, feature_size, 1, stride=1, padding=0)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.p4_conv2 = nn.Conv3d(feature_size, feature_size, 3, stride=1, padding=1)

        # Upsample C3 and add P4 to get P3
        self.p3_conv1 = nn.Conv3d(c3_size, feature_size, 1, stride=1, padding=0)
        self.p3_conv2 = nn.Conv3d(feature_size, feature_size, 3, stride=1, padding=1)

        self.p6_conv = nn.Conv3d(c5_size, feature_size, 3, stride=2, padding=1)
        self.p7_conv = nn.Conv3d(feature_size, feature_size, 3, stride=2, padding=1)

    def forward(self, c3, c4, c5):
        p5 = self.p5_conv1(c5)
        p5_u = self.p5_upsample(p5)
        p5 = self.p5_conv2(p5)

        p4 = self.p4_conv1(c4) + p5_u
        p4_u = self.p4_upsample(p4)
        p4 = self.p4_conv2(p4)

        p3 = self.p3_conv1(c3) + p4_u
        p3 = self.p3_conv2(p3)

        p6 = self.p6_conv(c5)
        p7 = self.p7_conv(p6.relu())

        return p3, p4, p5, p6, p7


class RegressionBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=256, num_anchors=15):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv3d(
            out_channels, num_anchors * 6, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        y = self.conv1(x).relu()
        y = self.conv2(y).relu()
        y = self.conv3(y).relu()
        y = self.conv4(y).relu()
        y = self.conv5(y)

        y = y.permute(0, 2, 3, 4, 1)

        return y.contiguous().view(y.shape[0], -1, 6)


class ClassificationBlock3D(nn.Module):
    def __init__(self, in_channels, feature_size=256, num_classes=10, num_anchors=15):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels, feature_size, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv3d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv3d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv3d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv3d(
            feature_size, num_anchors * num_classes, kernel_size=3, stride=1, padding=1
        )

        self.num_classes = num_classes
        self.num_anchors = num_anchors

    def forward(self, x):
        y = self.conv1(x).relu()
        y = self.conv2(y).relu()
        y = self.conv3(y).relu()
        y = self.conv4(y).relu()
        y = self.conv5(y)

        y = y.permute(0, 2, 3, 4, 1)
        b, w, h, d, _ = y.shape
        y = y.view(b, w, h, d, self.num_anchors, self.num_classes)

        return y.contiguous().view(y.shape[0], -1, self.num_classes)


class RetinaNet3D(nn.Module):
    def __init__(
        self,
        in_channels,
        feature_size=256,
        num_classes=1,
        num_anchors=15,
        backbone="resnet18",
    ):
        super().__init__()

        self.backbone = ResNet3D(in_channels, configuration=backbone)
        self.fpn = PyramidFeatures3D(128, 256, 512)

        self.regressgion_block = RegressionBlock3D(
            in_channels=256, out_channels=feature_size, num_anchors=num_anchors
        )

        self.classification_block = ClassificationBlock3D(
            in_channels=256,
            feature_size=feature_size,
            num_classes=num_classes + 2,
            num_anchors=num_anchors,
        )

        self.num_classes = num_classes + 2

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        features = self.fpn(c3, c4, c5)[:-2]
        box_y = torch.cat([self.regressgion_block(f) for f in features], dim=1)
        cls_y = torch.cat([self.classification_block(f) for f in features], dim=1)
        return box_y, cls_y


class FracNet(nn.Module):
    def __init__(self, in_channels, num_classes, first_out_channels=16):
        super().__init__()
        self.first = ConvBlock(in_channels, first_out_channels)
        in_channels = first_out_channels
        self.down1 = Down(in_channels, 2 * in_channels)
        self.down2 = Down(2 * in_channels, 4 * in_channels)
        self.down3 = Down(4 * in_channels, 8 * in_channels)
        self.up1   = Up(8 * in_channels, 4 * in_channels)
        self.up2   = Up(4 * in_channels, 2 * in_channels)
        self.up3   = Up(2 * in_channels, in_channels)
        self.final = nn.Conv3d(in_channels, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x1 = self.first(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x  = self.up1(x4, x3)
        x  = self.up2(x, x2)
        x  = self.up3(x, x1)
        x  = self.final(x)
        x = x.squeeze(1)
        x = torch.sigmoid(x)
        return x


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, y):
        x = self.conv2(x)
        x = self.conv1(torch.cat([y, x], dim=1))
        return x