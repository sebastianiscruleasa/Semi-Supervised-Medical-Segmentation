import torch
from torch import nn
from networks.myUnet import UpBlock


class RecurrentBlock(nn.Module):
    def __init__(self, channels, timestamps=2):
        super(RecurrentBlock, self).__init__()
        self.timestamps = timestamps
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        previous_state = self.layers(x)
        for i in range(self.timestamps):
            previous_state = self.layers(x + previous_state)
        return previous_state


class RecurrentResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, timestamps=2):
        super(RecurrentResidualBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.layers = nn.Sequential(
            RecurrentBlock(in_channels, out_channels, timestamps),
            RecurrentBlock(in_channels, out_channels, timestamps),
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.layers(x1)
        return x1 + x2


class RecurrentResidualDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, timestamps=2):
        super(RecurrentResidualDownBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            RecurrentResidualBlock(in_channels, out_channels, timestamps),
        )

    def forward(self, x):
        return self.layers(x)


class R2UNet(nn.Module):
    def __init__(self, in_channels, out_channels, timestamps=2):
        super(R2UNet, self).__init__()
        self.conv1 = RecurrentResidualBlock(in_channels, 64, timestamps)
        self.down1 = RecurrentResidualDownBlock(64, 128, timestamps)
        self.down2 = RecurrentResidualDownBlock(128, 256, timestamps)
        self.down3 = RecurrentResidualDownBlock(256, 512, timestamps)
        self.down4 = RecurrentResidualDownBlock(512, 1024, timestamps)
        self.up1 = UpBlock(1024, 512)
        self.conv2 = RecurrentResidualBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.conv3 = RecurrentResidualBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.conv4 = RecurrentResidualBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.conv5 = RecurrentResidualBlock(128, 64)
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv2(x)
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv3(x)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv4(x)
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv5(x)
        x = self.out(x)
        return x
