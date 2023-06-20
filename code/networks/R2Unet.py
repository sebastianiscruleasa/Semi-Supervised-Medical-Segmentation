import torch
from torch import nn
# from networks.Unet import UpBlock
from networks.NewAttentionUnet import UpConv as UpBlock


class RecurrentBlock(nn.Module):
    def __init__(self, channels, timestamps=2):
        super(RecurrentBlock, self).__init__()
        self.timestamps = timestamps
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU()
        )

    def forward(self, x):
        previous_state = self.layers(x)
        for i in range(1, self.timestamps):
            previous_state = self.layers(x + previous_state)
        return previous_state


class RecurrentResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, timestamps=2):
        super(RecurrentResidualConvBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.layers = nn.Sequential(
            RecurrentBlock(out_channels, timestamps),
            RecurrentBlock(out_channels, timestamps),
        )

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.layers(x1)
        return x1 + x2


class RecurrentResidualDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, timestamps=2):
        super(RecurrentResidualDownBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            RecurrentResidualConvBlock(in_channels, out_channels, timestamps),
        )

    def forward(self, x):
        return self.layers(x)


class R2UNet(nn.Module):
    def __init__(self, in_channels, out_channels, timestamps=2):
        super(R2UNet, self).__init__()
        self.conv1 = RecurrentResidualConvBlock(in_channels, 64, timestamps)

        self.down1 = RecurrentResidualDownBlock(64, 128, timestamps)
        self.down2 = RecurrentResidualDownBlock(128, 256, timestamps)
        self.down3 = RecurrentResidualDownBlock(256, 512, timestamps)
        self.down4 = RecurrentResidualDownBlock(512, 1024, timestamps)

        self.up1 = UpBlock(1024, 512)
        self.conv2 = RecurrentResidualConvBlock(1024, 512)

        self.up2 = UpBlock(512, 256)
        self.conv3 = RecurrentResidualConvBlock(512, 256)

        self.up3 = UpBlock(256, 128)
        self.conv4 = RecurrentResidualConvBlock(256, 128)

        self.up4 = UpBlock(128, 64)
        self.conv5 = RecurrentResidualConvBlock(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        encoder1 = self.conv1(x)

        encoder2 = self.down1(encoder1)
        encoder3 = self.down2(encoder2)
        encoder4 = self.down3(encoder3)
        encoder5 = self.down4(encoder4)

        decoder5 = self.up1(encoder5)
        decoder5 = torch.cat([encoder4, decoder5], dim=1)
        decoder5 = self.conv2(decoder5)

        decoder4 = self.up2(decoder5)
        decoder4 = torch.cat([encoder3, decoder4], dim=1)
        decoder4 = self.conv3(decoder4)

        decoder3 = self.up3(decoder4)
        decoder3 = torch.cat([encoder2, decoder3], dim=1)
        decoder3 = self.conv4(decoder3)

        decoder2 = self.up4(decoder3)
        decoder2 = torch.cat([encoder1, decoder2], dim=1)
        decoder2 = self.conv5(decoder2)

        decoder1 = self.out(decoder2)
        return decoder1
