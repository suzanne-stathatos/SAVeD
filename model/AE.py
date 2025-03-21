import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Upsample then Conv
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up_conv(x)



class AE_1024(nn.Module):
    def __init__(self, in_channels=3):
        super(AE_1024, self).__init__()
        print("AE_1024")
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.decoder = nn.Sequential(
            DecoderBlock(512, 256),
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 16),
            DecoderBlock(16, 1),
        )


    def forward(self, x):
        # x is concatenated input (B, 3, H, W)
        # Split into three frames
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        # print(f'x1 shape: {x1.shape}')
        # print(f'x2 shape: {x2.shape}')
        # print(f'x3 shape: {x3.shape}')
        x_encoded = self.encoder(x)
        # print(f'x_encoded shape: {x_encoded.shape}')
        x_decoded = self.decoder(x_encoded)
        # print(f'x_decoded shape: {x_decoded.shape}')
        return x_decoded