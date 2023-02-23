import torch
from torch import nn

from networks.myUnet import UpBlock, ConvBlock


class AttentionBlock(nn.Module):
    def __init__(self, g_channels, x_channels, int_channels):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_channels, int_channels, kernel_size=1),
            nn.BatchNorm2d(int_channels)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_channels, int_channels, kernel_size=1),
            nn.BatchNorm2d(int_channels)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionDownBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.layers(x)


class AttentionUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionUnet, self).__init__()
        self.conv1 = ConvBlock(in_channels, 64)
        self.down1 = AttentionDownBlock(64, 128)
        self.down2 = AttentionDownBlock(128, 256)
        self.down3 = AttentionDownBlock(256, 512)
        self.down4 = AttentionDownBlock(512, 1024)

        self.up1 = UpBlock(1024, 512)
        self.att1 = AttentionBlock(512, 512, 256)
        self.conv2 = ConvBlock(1024, 512)

        self.up2 = UpBlock(512, 256)
        self.att2 = AttentionBlock(256, 256, 128)
        self.conv3 = ConvBlock(512, 256)

        self.up3 = UpBlock(256, 128)
        self.att3 = AttentionBlock(128, 128, 64)
        self.conv4 = ConvBlock(256, 128)

        self.up4 = UpBlock(128, 64)
        self.att4 = AttentionBlock(64, 64, 32)
        self.conv5 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        encoder1 = self.conv1(x)
        encoder2 = self.down1(encoder1)
        encoder3 = self.down2(encoder2)
        encoder4 = self.down3(encoder3)
        encoder5 = self.down4(encoder4)

        decoder5 = self.up1(encoder5)
        attention4 = self.att1(decoder5, encoder4)
        decoder5 = torch.cat([attention4, decoder5], dim=1)
        decoder5 = self.conv2(decoder5)

        decoder4 = self.up2(decoder5)
        attention3 = self.att2(decoder4, encoder3)
        decoder4 = torch.cat([attention3, decoder4], dim=1)
        decoder4 = self.conv3(decoder4)

        decoder3 = self.up3(decoder4)
        attention2 = self.att3(decoder3, encoder2)
        decoder3 = torch.cat([attention2, decoder3], dim=1)
        decoder3 = self.conv4(decoder3)

        decoder2 = self.up4(decoder3)
        attention1 = self.att4(decoder2, encoder1)
        decoder2 = torch.cat([attention1, decoder2], dim=1)
        decoder2 = self.conv5(decoder2)

        d1 = self.out(decoder2)

        return d1

