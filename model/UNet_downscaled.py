'''
UNet_downscales model for denoising

Reduces the midchannels in the intermediate layers of the UNet to have the number of 
trainable parameters be roughly equal to those of our CNNs -- checked via metrics/count_params.ipynb
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


# From https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

class UNet_downscaled(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        
        # Three separate encoder paths
        self.encoder = nn.ModuleList([
            self._create_encoder(factor) for _ in range(3)
        ])
        
        # Combiner for bottleneck features
        self.combiner = nn.Sequential(
            DoubleConv(1024 * 3, 1024),  # Combine three 1024-channel features
            DoubleConv(1024, 1024 // factor)
        )
        
        # Decoder path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        self.combiner_512 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_256 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_128 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_64 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_32 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def _create_encoder(self, factor):
        return nn.ModuleList([
            DoubleConv(self.n_channels, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512),
            Down(512, 1024 // factor)
        ])
    
    
    def forward(self, x):
        # x is concatenated input (B, 3*C, H, W)
        # Split into three frames
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        # Lists to store skip connections
        skips1, skips2, skips3 = [], [], []
        
        # Encode each input
        x1_encoded = x1
        x2_encoded = x2
        x3_encoded = x3
        
        # Process each encoder level
        for i in range(len(self.encoder[0])):
            x1_encoded = self.encoder[0][i](x1_encoded)
            x2_encoded = self.encoder[1][i](x2_encoded)
            x3_encoded = self.encoder[2][i](x3_encoded)
            
            if i < len(self.encoder[0]) - 1:  # Store skip connections where newest is first
                skips1.insert(0, x1_encoded)
                skips2.insert(0, x2_encoded)
                skips3.insert(0, x3_encoded)
        
        # Combine encoded features and skip connections
        x = self.combiner_512(x1_encoded + x2_encoded + x3_encoded)
        skiped_combined0 = self.combiner_256(skips1[0] + skips2[0] + skips3[0] )
        skiped_combined1 = self.combiner_128(skips1[1] + skips2[1] + skips3[1] )
        skiped_combined2 = self.combiner_64(skips1[2] + skips2[2] + skips3[2] )
        skiped_combined3 = self.combiner_32(skips1[3] + skips2[3] + skips3[3] )
        
        # Decoder path with skip connections
        x = self.up1(x, skiped_combined0)
        x = self.up2(x, skiped_combined1)
        x = self.up3(x, skiped_combined2)
        x = self.up4(x, skiped_combined3)
        
        return self.outc(x)



# From https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # If mid_channels is not provided, set it to out_channels
        if not mid_channels:
            mid_channels = out_channels

        # Reduce mid_channels and out_channels by a factor of 16 to reduce the number of trainable parameters
        mid_channels = mid_channels // 16

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)