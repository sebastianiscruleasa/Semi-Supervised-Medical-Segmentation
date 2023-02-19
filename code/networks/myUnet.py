import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0):
        super(ConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(DownBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels, dropout),
        )

    def forward(self, x):
        return self.layers(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64, dropout=0.05)
        self.down1 = DownBlock(64, 128, dropout=0.1)
        self.down2 = DownBlock(128, 256, dropout=0.2)
        self.down3 = DownBlock(256, 512, dropout=0.3)
        self.down4 = DownBlock(512, 1024, dropout=0.5)
        self.up1 = UpBlock(1024, 512)
        self.conv2 = ConvBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.conv3 = ConvBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.conv4 = ConvBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.conv5 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = self.conv2(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.conv3(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.conv4(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.conv5(torch.cat([x, x1], dim=1))
        x = self.out(x)
        return x
