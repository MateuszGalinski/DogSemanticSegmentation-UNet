import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_conv1 = self.double_convolution(3, 64)
        self.encoder_conv2 = self.double_convolution(64, 128)
        self.encoder_conv3 = self.double_convolution(128, 256)
        self.encoder_conv4 = self.double_convolution(256, 512)
        self.encoder_conv5 = self.double_convolution(512, 1024)

        self.decoder_up_conv1 = self.up_transpose(1024, 512)
        self.decoder_conv1 = self.double_convolution(1024, 512)
        self.decoder_up_conv2 = self.up_transpose(512, 256)
        self.decoder_conv2 = self.double_convolution(512, 256)
        self.decoder_up_conv3 = self.up_transpose(256, 128)
        self.decoder_conv3 = self.double_convolution(256, 128)
        self.decoder_up_conv4 = self.up_transpose(128, 64)        
        self.decoder_conv4 = self.double_convolution(128, 64)
        
        self.out = nn.Conv2d(
            in_channels = 64,
            out_channels = num_classes,
            kernel_size = 1
        )

    def forward(self, x):
        down1 = self.encoder_conv1(x)
        down2 = self.max_pool2d(down1)
        down3 = self.encoder_conv2(down2)
        down4 = self.max_pool2d(down3)
        down5 = self.encoder_conv3(down4)
        down6 = self.max_pool2d(down5)
        down7 = self.encoder_conv4(down6)
        down8 = self.max_pool2d(down7)
        down9 = self.encoder_conv5(down8)

        up1 = self.decoder_up_conv1(down9)
        up2 = self.decoder_conv1(torch.cat([down7, up1], 1))

        up3 = self.decoder_up_conv2(up2)
        up4 = self.decoder_conv2(torch.cat([down5, up3], 1))

        up5 = self.decoder_up_conv3(up4)
        up6 = self.decoder_conv3(torch.cat([down3, up5], 1))

        up7 = self.decoder_up_conv4(up6)
        up8 = self.decoder_conv4(torch.cat([down1, up7], 1))

        out = self.out(up8)

        return out

    def double_convolution(self, in_channels, out_channels):
        double_conv_operation = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        return double_conv_operation
    
    def up_transpose(self, in_channels, out_channels):
        up_transpose_operation = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        return up_transpose_operation