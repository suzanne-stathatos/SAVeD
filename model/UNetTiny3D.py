# From Michael's use_dataset.ipynb notebook

# from unet_parts import *

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, conv_size=[3,3,3], padding=[1,1,1]):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                mid_channels,
                kernel_size=conv_size,
                padding=padding,
                bias=False,
            ),
            # nn.BatchNorm2d(mid_channels),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                mid_channels,
                out_channels,
                kernel_size=conv_size,
                padding=padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, max_pool=2, conv_size=[3,3,3], padding=[1,1,1]):
        super().__init__()
        # self.maxpool_conv = nn.Sequential(
        #     nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        # )
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(max_pool), DoubleConv3d(in_channels, out_channels, conv_size=conv_size, padding=padding)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=(2,2,2)):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.up = nn.Upsample(scale_factor=scale_factor, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3d(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv3d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # print(f"pre padd {x1.shape=} {x2.shape=}")

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]
        padd = (diffZ // 2, diffZ - diffZ // 2,diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2,)

        # [0, 0, 0, 0, 0, 1]
        #         Up-forward x1.shape=torch.Size([2, 512, 78, 6, 2])
        # Up-forward x2.shape=torch.Size([2, 512, 78, 6, 3])
        # [0, 0, 1]
        # [0, 0, 0, 0, 0, 1]
        # Up-forwardx2.shape=torch.Size([2, 512, 78, 6, 3]) x1.shape=torch.Size([2, 512, 79, 6, 2])
        x1 = F.pad(x1, padd)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        # print(f"padd {x.shape=}")
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=[1,1,1]):
        super(OutConv, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)


class UNetTiny3D(nn.Module):
    def __init__(self, n_channels, n_output_channels, n_timesteps_in=1, num_layers=4, start_hidden_dim=64, bilinear=False):
        super(UNetTiny3D, self).__init__()
        self.n_channels = n_channels
        self.n_output_channels = n_output_channels
        self.bilinear = bilinear
        self.n_timesteps_in = n_timesteps_in

        self.inc = DoubleConv3d(n_channels, start_hidden_dim)
        factor = 2 if bilinear else 1

        self.downs = nn.ModuleList()
        for i in range(num_layers):
            inp_d = start_hidden_dim*(2**i)
            op_d = start_hidden_dim*(2**(i+1))
            if i == (num_layers-1):
                op_d = op_d//factor
            self.downs.append(Down(inp_d, op_d))

        # factor = 2 if bilinear else 1
        # self.down4 = Down(start_hidden_dim*(2**num_layers-1), start_hidden_dim*(2**num_layers) // factor)

        self.ups = nn.ModuleList()
        for i in range(num_layers,1,-1):
            inp_d = start_hidden_dim * (2**i)
            op_d = start_hidden_dim * (2 ** (i - 1) // factor)
            self.ups.append(
                Up(
                    inp_d,
                    op_d,
                    bilinear,
                )
            )
        self.ups.append(
                Up(
                    2*start_hidden_dim, start_hidden_dim, bilinear
                )
            )
        # self.up4 = Up(2*start_hidden_dim, start_hidden_dim, bilinear)


            # self.up1 = Up(1024, 512 // factor, bilinear)

        # self.down1 = Down(64, 128)
        # self.down2 = Down(128, 256)
        # self.down3 = Down(256, 512)
        # self.down4 = Down(512, 1024 // factor)
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # # self.down4 = Down(512, 1024 // factor, max_pool=(2,2,1), conv_size=[3,3,1], padding=[1,1,0])
        # # self.up1 = Up(1024, 512 // factor, bilinear, scale_factor=(2, 2, 1))
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(2*start_hidden_dim, start_hidden_dim, bilinear)
        self.outc = OutConv(start_hidden_dim, n_output_channels, kernel_size=[1,1,n_timesteps_in])


    def forward(self, x):
        xs_down = []
        # print(f"{x.shape=}")
        xs_down.append(self.inc(x))
        for down_layer in self.downs:
            xs_down.append(down_layer(xs_down[-1]))
        x_prev = xs_down[-1]
        xs_down = xs_down[:-1]
        for xs_corresponding, up_layer in zip(xs_down[::-1], self.ups):
            op = up_layer(x_prev, xs_corresponding)
            x_prev = op
        op = self.outc(x_prev)
        return op

        # x1 = self.inc(x)
        # # print(f"{x1.shape=}")
        # x2 = self.down1(x1)
        # # print(f"{x2.shape=}")
        # x3 = self.down2(x2)
        # # print(f"{x3.shape=}")
        # x4 = self.down3(x3)
        # # print(f"{x4.shape=}")
        # x5 = self.down4(x4)
        # print(f"{x5.shape=}")
        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # logits = self.outc(x)
        # return logits