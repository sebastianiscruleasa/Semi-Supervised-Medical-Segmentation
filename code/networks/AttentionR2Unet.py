import torch
from torch import nn
from networks.Unet import UpBlock, ConvBlock
from networks.R2Unet import RecurrentResidualConvBlock, RecurrentResidualDownBlock
# from networks.AttentionUnet import AttentionBlock
from networks.NewAttentionUnet import AttentionBlock


class AttentionR2Unet(nn.Module):
    def __init__(self, in_channels, out_channels, timestamps=2):
        super(AttentionR2Unet, self).__init__()
        self.conv1 = RecurrentResidualConvBlock(in_channels, 64, timestamps)

        self.down1 = RecurrentResidualDownBlock(64, 128, timestamps)
        self.down2 = RecurrentResidualDownBlock(128, 256, timestamps)
        self.down3 = RecurrentResidualDownBlock(256, 512, timestamps)
        self.down4 = RecurrentResidualDownBlock(512, 1024, timestamps)

        self.up1 = UpBlock(1024, 512)
        self.att1 = AttentionBlock(512, 512, 256)
        self.conv2 = RecurrentResidualConvBlock(1024, 512)

        self.up2 = UpBlock(512, 256)
        self.att2 = AttentionBlock(256, 256, 128)
        self.conv3 = RecurrentResidualConvBlock(512, 256)

        self.up3 = UpBlock(256, 128)
        self.att3 = AttentionBlock(128, 128, 64)
        self.conv4 = RecurrentResidualConvBlock(256, 128)

        self.up4 = UpBlock(128, 64)
        self.att4 = AttentionBlock(64, 64, 32)
        self.conv5 = RecurrentResidualConvBlock(128, 64)

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

        decoder1 = self.out(decoder2)
        return decoder1
