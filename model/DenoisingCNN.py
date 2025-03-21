import torch
import torch.nn as nn

class DenoisingCNN_512(nn.Module):
    def __init__(self, fine_layers, bottleneck_size):
        super(DenoisingCNN_512, self).__init__()
        print("DenoisingCNN_512")
        self.fine_layers = fine_layers
        self.bottleneck_size = bottleneck_size
        if self.fine_layers:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
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
            )

            self.future_decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                # TODO: try ablating the last layer but add an additional filter in the middle -- fine layers
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

            self.future_decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            )
        
        # Play with compressed hyperparams
        self.combiner_64 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_32 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_16 = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        # x is concatenated input (B, 3*C, H, W)
        # Split into three frames
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x3 = self.encoder(x3)
        if self.bottleneck_size == 16:
            x_comb = self.combiner_16(x1 + x2 + x3)
        elif self.bottleneck_size == 32:
            x_comb = self.combiner_32(x1 + x2 + x3)
        elif self.bottleneck_size == 64:
            x_comb = self.combiner_64(x1 + x2 + x3)
        x_decoded = self.future_decoder(x_comb)
        return x_decoded



class DenoisingCNN_512_with_skips(DenoisingCNN_512):
    def __init__(self, fine_layers, bottleneck_size):
        fine_layers = True  # fine_layers = True by default
        super(DenoisingCNN_512_with_skips, self).__init__(fine_layers, bottleneck_size)
        print("DenoisingCNN_512_with_skips")

        # Encoder blocks with internal skip connections
        self.enc1_conv = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc1_pool = nn.MaxPool2d(2)
        self.enc2_conv = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc2_pool = nn.MaxPool2d(2)
        self.enc3_conv = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc3_pool = nn.MaxPool2d(2)
        self.enc4_conv = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc4_pool = nn.MaxPool2d(2)
        self.enc5_conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc5_pool = nn.MaxPool2d(2)

        # Skip connections with pooling
        self.enc1_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, kernel_size=1)
        )
        self.enc2_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=1)
        )
        self.enc3_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=1)
        )
        self.enc4_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.enc5_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=1)
        )

        # Feature combiners
        self.feature_combiners = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c*3, c, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.ReLU()
            ) for c in [16, 32, 64, 128, 256]
        ])

        # Decoder with correct channel sizes after concatenation
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),  # 128 + 256 = 384
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),  # 64 + 128 = 192
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),  # 32 + 64 = 96
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=3, padding=1),  # 16 + 32 = 48
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        )

    def encoder(self, x):
        """Encoder function that returns all intermediate features"""
        features = []
        
        # Encoder block 1
        x1 = self.enc1_conv(x)
        features.append(x1)
        x1_pool = self.enc1_pool(x1)
        x1 = x1_pool + self.enc1_skip(x1)
        
        # Encoder block 2
        x2 = self.enc2_conv(x1)
        features.append(x2)
        x2_pool = self.enc2_pool(x2)
        x2 = x2_pool + self.enc2_skip(x2)
        
        # Encoder block 3
        x3 = self.enc3_conv(x2)
        features.append(x3)
        x3_pool = self.enc3_pool(x3)
        x3 = x3_pool + self.enc3_skip(x3)
        
        # Encoder block 4
        x4 = self.enc4_conv(x3)
        features.append(x4)
        x4_pool = self.enc4_pool(x4)
        x4 = x4_pool + self.enc4_skip(x4)
        
        # Encoder block 5
        x5 = self.enc5_conv(x4)
        features.append(x5)
        x5_pool = self.enc5_pool(x5)
        x5 = x5_pool + self.enc5_skip(x5)
        
        return x5, features

    def future_decoder(self, x, features):
        """Decoder function that uses stored features for skip connections"""
        features = features[::-1]
        
        # Decoder block 1
        d1 = self.dec1(x)
        d1 = torch.cat([d1, features[0]], dim=1)
        
        # Decoder block 2
        d2 = self.dec2(d1)
        d2 = torch.cat([d2, features[1]], dim=1)
        
        # Decoder block 3
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, features[2]], dim=1)
        
        # Decoder block 4
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, features[3]], dim=1)
        
        # Decoder block 5 (final block)
        out = self.dec5(d4)
        
        return out

    def forward(self, x):
        # Split into three frames
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        
        # Get encoded representations and features
        x1, features1 = self.encoder(x1)
        x2, features2 = self.encoder(x2)
        x3, features3 = self.encoder(x3)
        
        # Combine encoded representations
        if self.bottleneck_size == 256:
            x_comb = self.combiner_256(x1 + x2 + x3)
        elif self.bottleneck_size == 128:
            x_comb = self.combiner_128(x1 + x2 + x3)
        elif self.bottleneck_size == 64:
            x_comb = self.combiner_64(x1 + x2 + x3)
        elif self.bottleneck_size == 32:
            x_comb = self.combiner_32(x1 + x2 + x3)
        elif self.bottleneck_size == 16:
            x_comb = self.combiner_16(x1 + x2 + x3)
        else:
            raise ValueError(f"Invalid bottleneck size: {self.bottleneck_size}")
        
        # Combine features using dedicated combiners
        combined_features = []
        for idx, (f1, f2, f3) in enumerate(zip(features1, features2, features3)):
            combined = torch.cat([f1, f2, f3], dim=1)
            combined_features.append(self.feature_combiners[idx](combined))
        
        # Decode with combined features
        return self.future_decoder(x_comb, combined_features)        


class DenoisingCNN_1024(nn.Module):
    def __init__(self, fine_layers, bottleneck_size):
        super(DenoisingCNN_1024, self).__init__()
        print("DenoisingCNN_1024")
        self.fine_layers = fine_layers
        self.bottleneck_size = bottleneck_size

        if self.fine_layers:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
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
            self.future_decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
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

            self.future_decoder = nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            )

        self.combiner_16 = nn.Sequential(
            nn.Conv2d(512, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_32 = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_64 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_128 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_256 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_512 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )


    def forward(self, x):
        # x is concatenated input (B, 3*C, H, W)
        # Split into three frames
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x3 = self.encoder(x3)
        if self.bottleneck_size == 16:
            x_comb = self.combiner_16(x1 + x2 + x3)
        elif self.bottleneck_size == 32:
            x_comb = self.combiner_32(x1 + x2 + x3)
        elif self.bottleneck_size == 64:
            x_comb = self.combiner_64(x1 + x2 + x3)
        elif self.bottleneck_size == 128:
            x_comb = self.combiner_128(x1 + x2 + x3)
        elif self.bottleneck_size == 256:
            x_comb = self.combiner_256(x1 + x2 + x3)
        elif self.bottleneck_size == 512:
            x_comb = self.combiner_512(x1 + x2 + x3)
        x_decoded = self.future_decoder(x_comb)
        return x_decoded


class DenoisingCNN_1024_with_skips(DenoisingCNN_1024):
    def __init__(self, fine_layers, bottleneck_size):
        fine_layers = True # fine_layers = True by default
        super(DenoisingCNN_1024_with_skips, self).__init__(fine_layers, bottleneck_size) # inherits the combiner layers
        print("DenoisingCNN_1024_with_skips")

        # Encoder blocks with internal skip connections
        self.enc1_conv = nn.Sequential( nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc1_pool = nn.MaxPool2d(2)
        self.enc2_conv = nn.Sequential( nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc2_pool = nn.MaxPool2d(2)
        self.enc3_conv = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc3_pool = nn.MaxPool2d(2)
        self.enc4_conv = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc4_pool = nn.MaxPool2d(2)
        self.enc5_conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc5_pool = nn.MaxPool2d(2)
        self.enc6_conv = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc6_pool = nn.MaxPool2d(2)

        # Additional conv layers for encoder skip connections
        self.enc1_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, kernel_size=1)
        )
        self.enc2_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=1)
        )
        self.enc3_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=1)
        )
        self.enc4_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.enc5_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=1)
        )
        self.enc6_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=1)
        )

        # Decoder with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, padding=1),  # 256 + 512 = 768
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 256 + 256 = 512
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 128 + 128 = 256
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # 64 + 64 = 128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec6 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),  # 32 + 32 = 64
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        )

        self.feature_combiners = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c*3, c, kernel_size=3, padding=1),  # 3x channels because we concat 3 frames
                nn.ReLU(),
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.ReLU()
            ) for c in [16, 32, 64, 128, 256, 512]  # Added 512
        ])


    def encoder(self,x):
        """Encoder function that returns all intermediate features"""
        features = []
        # print('In Encoder')
        # print(x.shape)
        # Encoder with internal skip connections
        e1_conv = self.enc1_conv(x)
        # print(f'conv1: {e1_conv.shape}')
        e1_pool = self.enc1_pool(e1_conv)
        # print(f'pool1: {e1_pool.shape}')
        e1 = e1_pool + self.enc1_skip(e1_conv)
        # print(f'skip1: {e1.shape}')
        features.append(e1_conv)  # Store pre-pool features for decoder
        e2_conv = self.enc2_conv(e1)
        # print(f'conv2: {e2_conv.shape}')
        e2_pool = self.enc2_pool(e2_conv)
        # print(f'pool2: {e2_pool.shape}')
        e2 = e2_pool + self.enc2_skip(e2_conv)
        # print(f'skip2: {e2.shape}')
        features.append(e2_conv)
        e3_conv = self.enc3_conv(e2)
        # print(f'conv3: {e3_conv.shape}')
        e3_pool = self.enc3_pool(e3_conv)
        # print(f'pool3: {e3_pool.shape}')
        e3 = e3_pool + self.enc3_skip(e3_conv)
        # print(f'skip3: {e3.shape}')
        features.append(e3_conv)
        e4_conv = self.enc4_conv(e3)
        # print(f'conv4: {e4_conv.shape}')
        e4_pool = self.enc4_pool(e4_conv)
        # print(f'pool4: {e4_pool.shape}')
        e4 = e4_pool + self.enc4_skip(e4_conv)
        # print(f'skip4: {e4.shape}')
        features.append(e4_conv)
        e5_conv = self.enc5_conv(e4)
        # print(f'conv5: {e5_conv.shape}')
        e5_pool = self.enc5_pool(e5_conv)
        # print(f'pool5: {e5_pool.shape}')
        e5 = e5_pool + self.enc5_skip(e5_conv)
        # print(f'skip5: {e5.shape}')
        features.append(e5_conv)
        e6_conv = self.enc6_conv(e5)
        # print(f'conv6: {e6_conv.shape}')
        e6_pool = self.enc6_pool(e6_conv)
        # print(f'pool6: {e6_pool.shape}')
        e6 = e6_pool + self.enc6_skip(e6_conv)
        # print(f'skip6: {e6.shape}')
        features.append(e6_conv)
        return e6, features

    
    def future_decoder(self, x, features):
        """Decoder function that uses stored features for skip connections"""
        # Reverse features list for easier indexing
        features = features[::-1]
        # print('In Future Decoder')
        # print(x.shape)
        # Decoder with skip connections
        d1 = self.dec1(x)
        # print(f'dec1: {d1.shape}')
        d1 = torch.cat([d1, features[0]], dim=1)
        # print(f'd1: {d1.shape}')
        d2 = self.dec2(d1)
        # print(f'dec2: {d2.shape}')
        d2 = torch.cat([d2, features[1]], dim=1)
        # print(f'd2: {d2.shape}')
        d3 = self.dec3(d2)
        # print(f'dec3: {d3.shape}')
        d3 = torch.cat([d3, features[2]], dim=1)
        # print(f'd3: {d3.shape}')
        d4 = self.dec4(d3)
        # print(f'dec4: {d4.shape}')
        d4 = torch.cat([d4, features[3]], dim=1)
        # print(f'd4: {d4.shape}')
        d5 = self.dec5(d4)
        # print(f'dec5: {d5.shape}')
        d5 = torch.cat([d5, features[4]], dim=1)
        # print(f'd5: {d5.shape}')
        out = self.dec6(d5)
        # print(f'out: {out.shape}')
        return out


    def forward(self, x):
        # x is concatenated input (B, 3*C, H, W)
        # Split into three frames
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        # Get encoded representations and features
        x1, features1 = self.encoder(x1)
        x2, features2 = self.encoder(x2)
        x3, features3 = self.encoder(x3)
        # Combine only the encoded representations, not the features
        if self.bottleneck_size == 16:
            x_comb = self.combiner_16(x1 + x2 + x3)
        elif self.bottleneck_size == 32:
            x_comb = self.combiner_32(x1 + x2 + x3)
        elif self.bottleneck_size == 64:
            x_comb = self.combiner_64(x1 + x2 + x3)
        elif self.bottleneck_size == 128:
            x_comb = self.combiner_128(x1 + x2 + x3)
        elif self.bottleneck_size == 256:
            x_comb = self.combiner_256(x1 + x2 + x3)
        elif self.bottleneck_size == 512:
            # print('In Combiner 512')
            # print(x1.shape)
            # print(x2.shape)
            # print(x3.shape)
            x_comb = self.combiner_512(x1 + x2 + x3)
            # print(f'x_comb: {x_comb.shape}')
        # Combine features as well
        # Combine features using dedicated combiners
        combined_features = []
        for idx, (f1, f2, f3) in enumerate(zip(features1, features2, features3)):
            # Concatenate features from all frames
            combined = torch.cat([f1, f2, f3], dim=1)
            # Apply feature combiner
            combined_features.append(self.feature_combiners[idx](combined))

        # Decode with combined features
        return self.future_decoder(x_comb, combined_features)


class DenoisingCNN_1024_with_residual_connections(DenoisingCNN_1024):
    def __init__(self, fine_layers, bottleneck_size):
        super(DenoisingCNN_1024_with_residual_connections, self).__init__(fine_layers, bottleneck_size)
        print("DenoisingCNN_1024_with_residual_connections")
        # Encoder blocks with internal skip connections and residual connections
        self.enc1_conv = nn.ModuleDict({
            'conv1': nn.Conv2d(1, 16, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(16, 16, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })
        self.enc1_pool = nn.MaxPool2d(2)
        self.enc2_conv = nn.ModuleDict({
            'conv1': nn.Conv2d(16, 32, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(32, 32, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })
        self.enc2_pool = nn.MaxPool2d(2)
        self.enc3_conv = nn.ModuleDict({
            'conv1': nn.Conv2d(32, 64, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(64, 64, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })
        self.enc3_pool = nn.MaxPool2d(2)
        self.enc4_conv = nn.ModuleDict({
            'conv1': nn.Conv2d(64, 128, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(128, 128, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })
        self.enc4_pool = nn.MaxPool2d(2)
        self.enc5_conv = nn.ModuleDict({
            'conv1': nn.Conv2d(128, 256, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })
        self.enc5_pool = nn.MaxPool2d(2)
        self.enc6_conv = nn.ModuleDict({
            'conv1': nn.Conv2d(256, 512, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })
        self.enc6_pool = nn.MaxPool2d(2)

        # Enhanced skip connections with residual blocks
        self.enc1_skip = nn.ModuleDict({
            'pool': nn.MaxPool2d(2),
            'conv1': nn.Conv2d(16, 16, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(16, 16, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })
        
        self.enc2_skip = nn.ModuleDict({
            'pool': nn.MaxPool2d(2),
            'conv1': nn.Conv2d(32, 32, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(32, 32, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })
        
        self.enc3_skip = nn.ModuleDict({
            'pool': nn.MaxPool2d(2),
            'conv1': nn.Conv2d(64, 64, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(64, 64, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })
        
        self.enc4_skip = nn.ModuleDict({
            'pool': nn.MaxPool2d(2),
            'conv1': nn.Conv2d(128, 128, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(128, 128, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })
        
        self.enc5_skip = nn.ModuleDict({
            'pool': nn.MaxPool2d(2),
            'conv1': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })
        
        self.enc6_skip = nn.ModuleDict({
            'pool': nn.MaxPool2d(2),
            'conv1': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'relu2': nn.ReLU()
        })

        # Decoder with skip connections and residual connections
        self.dec1 = nn.ModuleDict({
            'conv1': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'relu2': nn.ReLU(),
            'upsample': nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        })
        self.dec2 = nn.ModuleDict({
            'conv1': nn.Conv2d(768, 512, kernel_size=3, padding=1),  # 256 + 512 = 768
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(512, 512, kernel_size=3, padding=1),
            'relu2': nn.ReLU(),
            'upsample': nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        })
        self.dec3 = nn.ModuleDict({
            'conv1': nn.Conv2d(512, 256, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'relu2': nn.ReLU(),
            'upsample': nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        })
        self.dec4 = nn.ModuleDict({
            'conv1': nn.Conv2d(256, 128, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(128, 128, kernel_size=3, padding=1),
            'relu2': nn.ReLU(),
            'upsample': nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        })
        self.dec5 = nn.ModuleDict({
            'conv1': nn.Conv2d(128, 64, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(64, 64, kernel_size=3, padding=1),
            'relu2': nn.ReLU(),
            'upsample': nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        })
        self.dec6 = nn.ModuleDict({
            'conv1': nn.Conv2d(64, 32, kernel_size=3, padding=1),
            'relu1': nn.ReLU(),
            'conv2': nn.Conv2d(32, 32, kernel_size=3, padding=1),
            'relu2': nn.ReLU(),
            'upsample': nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        })

        self.feature_combiners = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c*3, c, kernel_size=3, padding=1),  # 3x channels because we concat 3 frames
                nn.ReLU(),
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.ReLU()
            ) for c in [16, 32, 64, 128, 256, 512]  # Added 512
        ])

        self.dim_matches = nn.ModuleDict({
            '1_16': nn.Conv2d(1, 16, 1),
            '16_32': nn.Conv2d(16, 32, 1),
            '32_64': nn.Conv2d(32, 64, 1),
            '64_128': nn.Conv2d(64, 128, 1),
            '128_256': nn.Conv2d(128, 256, 1),
            '256_512': nn.Conv2d(256, 512, 1)
        })

    def residual_block(self, x, block):
        """Helper function to apply residual connection"""
        identity = x
        out = block['conv1'](x)
        out = block['relu1'](out)
        out = block['conv2'](out)
        
        # If dimensions don't match, transform identity
        if identity.shape[1] != out.shape[1]:
            in_channels = identity.shape[1]
            out_channels = out.shape[1]
            key = f'{in_channels}_{out_channels}'
            identity = self.dim_matches[key](identity)
            
        out += identity
        return block['relu2'](out)

    def apply_skip_connection(self, x, skip_block):
        """Helper function to apply skip connection with residual block"""
        x = skip_block['pool'](x)
        identity = x
        
        out = skip_block['conv1'](x)
        out = skip_block['relu1'](out)
        out = skip_block['conv2'](out)
        
        # Add residual connection
        out += identity
        return skip_block['relu2'](out)


    def encoder(self,x):
        """Encoder function that returns all intermediate features"""
        features = []
        # Encoder with internal skip connections and residual blocks in encoder
        e1_conv = self.residual_block(x, self.enc1_conv)
        e1_pool = self.enc1_pool(e1_conv)
        e1 = e1_pool + self.apply_skip_connection(e1_conv, self.enc1_skip)
        features.append(e1_conv) # Store pre-pool features for decoder
        
        e2_conv = self.residual_block(e1_pool, self.enc2_conv)
        e2_pool = self.enc2_pool(e2_conv)
        e2 = e2_pool + self.apply_skip_connection(e2_conv, self.enc2_skip)
        features.append(e2_conv)

        e3_conv = self.residual_block(e2_pool, self.enc3_conv)
        e3_pool = self.enc3_pool(e3_conv)
        e3 = e3_pool + self.apply_skip_connection(e3_conv, self.enc3_skip)
        features.append(e3_conv)

        e4_conv = self.residual_block(e3_pool, self.enc4_conv)
        e4_pool = self.enc4_pool(e4_conv)
        e4 = e4_pool + self.apply_skip_connection(e4_conv, self.enc4_skip)
        features.append(e4_conv)

        e5_conv = self.residual_block(e4_pool, self.enc5_conv)
        e5_pool = self.enc5_pool(e5_conv)
        e5 = e5_pool + self.apply_skip_connection(e5_conv, self.enc5_skip)
        features.append(e5_conv)

        e6_conv = self.residual_block(e5_pool, self.enc6_conv)
        e6_pool = self.enc6_pool(e6_conv)
        e6 = e6_pool + self.apply_skip_connection(e6_conv, self.enc6_skip)
        features.append(e6_conv)
        return e6, features

    
    def future_decoder(self, x, features):
        """Decoder function that uses stored features for skip connections"""
        # Reverse features list for easier indexing
        features = features[::-1]
        # Decoder with skip connections and residual connections
        identity = x
        d1 = self.dec1['conv1'](x)
        d1 = self.dec1['relu1'](d1)
        d1 = self.dec1['conv2'](d1)
        if identity.shape[1] == d1.shape[1]:
            d1 += identity
        d1 = self.dec1['relu2'](d1)
        d1 = self.dec1['upsample'](d1)
        # Apply skip connection by adding the features
        d1 = torch.cat([d1, features[0]], dim=1)
        d2 = self.dec2['conv1'](d1)
        d2 = self.dec2['relu1'](d2)
        d2 = self.dec2['conv2'](d2)
        if d1.shape[1] == d2.shape[1]:
            d2 += d1
        d2 = self.dec2['relu2'](d2)
        d2 = self.dec2['upsample'](d2)
        d2 = torch.cat([d2, features[1]], dim=1)
        d3 = self.dec3['conv1'](d2)
        d3 = self.dec3['relu1'](d3)
        d3 = self.dec3['conv2'](d3)
        if d2.shape[1] == d3.shape[1]:
            d3 += d2
        d3 = self.dec3['relu2'](d3)
        d3 = self.dec3['upsample'](d3)
        d3 = torch.cat([d3, features[2]], dim=1)
        d4 = self.dec4['conv1'](d3)
        d4 = self.dec4['relu1'](d4)
        d4 = self.dec4['conv2'](d4)
        if d3.shape[1] == d4.shape[1]:
            d4 += d3
        d4 = self.dec4['relu2'](d4)
        d4 = self.dec4['upsample'](d4)
        d4 = torch.cat([d4, features[3]], dim=1)
        d5 = self.dec5['conv1'](d4)
        d5 = self.dec5['relu1'](d5)
        d5 = self.dec5['conv2'](d5)
        if d4.shape[1] == d5.shape[1]:
            d5 += d4
        d5 = self.dec5['relu2'](d5)
        d5 = self.dec5['upsample'](d5)
        d5 = torch.cat([d5, features[4]], dim=1)
        d6 = self.dec6['conv1'](d5)
        d6 = self.dec6['relu1'](d6)
        d6 = self.dec6['conv2'](d6)
        if d5.shape[1] == d6.shape[1]:
            d6 += d5
        d6 = self.dec6['relu2'](d6)
        out = self.dec6['upsample'](d6)
        return out


    def forward(self, x):
        # x is concatenated input (B, 3*C, H, W)
        # Split into three frames
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        # Get encoded representations and features
        x1, features1 = self.encoder(x1)
        x2, features2 = self.encoder(x2)
        x3, features3 = self.encoder(x3)
        # Combine only the encoded representations, not the features
        if self.bottleneck_size == 16:
            x_comb = self.combiner_16(x1 + x2 + x3)
        elif self.bottleneck_size == 32:
            x_comb = self.combiner_32(x1 + x2 + x3)
        elif self.bottleneck_size == 64:
            x_comb = self.combiner_64(x1 + x2 + x3)
        elif self.bottleneck_size == 128:
            x_comb = self.combiner_128(x1 + x2 + x3)
        elif self.bottleneck_size == 256:
            x_comb = self.combiner_256(x1 + x2 + x3)
        elif self.bottleneck_size == 512:
            x_comb = self.combiner_512(x1 + x2 + x3)
        # Combine features as well
        # Combine features using dedicated combiners
        combined_features = []
        for idx, (f1, f2, f3) in enumerate(zip(features1, features2, features3)):
            # Concatenate features from all frames
            combined = torch.cat([f1, f2, f3], dim=1)
            # Apply feature combiner
            combined_features.append(self.feature_combiners[idx](combined))

        # Decode with combined features
        return self.future_decoder(x_comb, combined_features)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        
        # Dimension matching layer if needed
        self.dim_match = None
        if in_channels != out_channels:
            self.dim_match = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Handle dimension matching
        if self.dim_match is not None:
            identity = self.dim_match(identity)
            
        # Add residual connection
        out += identity
        return self.relu2(out)


class DenoisingCNN_1024_with_skips_and_resnet_blocks(DenoisingCNN_1024):
    def __init__(self, fine_layers, bottleneck_size):
        fine_layers = True # fine_layers = True by default
        super(DenoisingCNN_1024_with_skips_and_resnet_blocks, self).__init__(fine_layers, bottleneck_size) 
        print("DenoisingCNN_1024_with_skips_and_resnet_blocks")

        # Encoder ResNet blocks
        self.enc1_conv = ResNetBlock(1, 16)
        self.enc1_pool = nn.MaxPool2d(2)
        self.enc2_conv = ResNetBlock(16, 32)
        self.enc2_pool = nn.MaxPool2d(2)
        self.enc3_conv = ResNetBlock(32, 64)
        self.enc3_pool = nn.MaxPool2d(2)
        self.enc4_conv = ResNetBlock(64, 128)
        self.enc4_pool = nn.MaxPool2d(2)
        self.enc5_conv = ResNetBlock(128, 256)
        self.enc5_pool = nn.MaxPool2d(2)
        self.enc6_conv = ResNetBlock(256, 512)
        self.enc6_pool = nn.MaxPool2d(2)
        
        # Skip connection ResNet blocks
        self.enc1_skip = nn.Sequential(
            nn.MaxPool2d(2),
            ResNetBlock(16, 16)
        )
        self.enc2_skip = nn.Sequential(
            nn.MaxPool2d(2),
            ResNetBlock(32, 32)
        )
        self.enc3_skip = nn.Sequential(
            nn.MaxPool2d(2),
            ResNetBlock(64, 64)
        )
        self.enc4_skip = nn.Sequential(
            nn.MaxPool2d(2),
            ResNetBlock(128, 128)
        )
        self.enc5_skip = nn.Sequential(
            nn.MaxPool2d(2),
            ResNetBlock(256, 256)
        )
        self.enc6_skip = nn.Sequential(
            nn.MaxPool2d(2),
            ResNetBlock(512, 512)
        )
        
        # Decoder blocks
        self.dec1 = nn.ModuleDict({
            'resnet': ResNetBlock(512, 512),
            'upsample': nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        })
        
        self.dec2 = nn.ModuleDict({
            'resnet': ResNetBlock(768, 512),  # 256 + 512 = 768
            'upsample': nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        })
        
        self.dec3 = nn.ModuleDict({
            'resnet': ResNetBlock(512, 256),  # 256 + 256 = 512
            'upsample': nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        })
        
        self.dec4 = nn.ModuleDict({
            'resnet': ResNetBlock(256, 128),  # 128 + 128 = 256
            'upsample': nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        })
        
        self.dec5 = nn.ModuleDict({
            'resnet': ResNetBlock(128, 64),  # 64 + 64 = 128
            'upsample': nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        })
        
        self.dec6 = nn.ModuleDict({
            'resnet': ResNetBlock(64, 32),  # 32 + 32 = 64
            'upsample': nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        })
        
        self.feature_combiners = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c*3, c, kernel_size=1),
                nn.BatchNorm2d(c),
                ResNetBlock(c, c)
            ) for c in [16, 32, 64, 128, 256, 512]
        ])


    def encoder(self,x):
        """Encoder function that returns all intermediate features"""
        features = []
        # Encoder with internal skip connections
        e1_conv = self.enc1_conv(x)
        e1_pool = self.enc1_pool(e1_conv)
        e1 = e1_pool + self.enc1_skip(e1_conv)
        features.append(e1_conv)
        e2_conv = self.enc2_conv(e1)
        e2_pool = self.enc2_pool(e2_conv)
        e2 = e2_pool + self.enc2_skip(e2_conv)
        features.append(e2_conv)
        e3_conv = self.enc3_conv(e2)
        e3_pool = self.enc3_pool(e3_conv)
        e3 = e3_pool + self.enc3_skip(e3_conv)
        features.append(e3_conv)
        e4_conv = self.enc4_conv(e3)
        e4_pool = self.enc4_pool(e4_conv)
        e4 = e4_pool + self.enc4_skip(e4_conv)
        features.append(e4_conv)
        e5_conv = self.enc5_conv(e4)
        e5_pool = self.enc5_pool(e5_conv)
        e5 = e5_pool + self.enc5_skip(e5_conv)
        features.append(e5_conv)
        e6_conv = self.enc6_conv(e5)
        e6_pool = self.enc6_pool(e6_conv)
        e6 = e6_pool + self.enc6_skip(e6_conv)
        features.append(e6_conv)
        return e6, features

    
    def future_decoder(self, x, features):
        """Decoder function that uses stored features for skip connections"""
        # Reverse features list for easier indexing
        features = features[::-1]
        # Decoder with skip connections
        d1 = self.dec1['resnet'](x)
        d1 = self.dec1['upsample'](d1)
        d1 = torch.cat([d1, features[0]], dim=1)
        d2 = self.dec2['resnet'](d1)
        d2 = self.dec2['upsample'](d2)
        d2 = torch.cat([d2, features[1]], dim=1)
        d3 = self.dec3['resnet'](d2)
        d3 = self.dec3['upsample'](d3)
        d3 = torch.cat([d3, features[2]], dim=1)
        d4 = self.dec4['resnet'](d3)
        d4 = self.dec4['upsample'](d4)
        d4 = torch.cat([d4, features[3]], dim=1)
        d5 = self.dec5['resnet'](d4)
        d5 = self.dec5['upsample'](d5)
        d5 = torch.cat([d5, features[4]], dim=1)
        d6 = self.dec6['resnet'](d5)
        out = self.dec6['upsample'](d6)
        return out


    def forward(self, x):
        # x is concatenated input (B, 3*C, H, W)
        # Split into three frames
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        
        # Get encoded representations and features
        x1, features1 = self.encoder(x1)
        x2, features2 = self.encoder(x2)
        x3, features3 = self.encoder(x3)
        
        # Combine encoded representations
        if self.bottleneck_size == 16:
            x_comb = self.combiner_16(x1 + x2 + x3)
        elif self.bottleneck_size == 32:
            x_comb = self.combiner_32(x1 + x2 + x3)
        elif self.bottleneck_size == 64:
            x_comb = self.combiner_64(x1 + x2 + x3)
        elif self.bottleneck_size == 128:
            x_comb = self.combiner_128(x1 + x2 + x3)
        elif self.bottleneck_size == 256:
            x_comb = self.combiner_256(x1 + x2 + x3)
        elif self.bottleneck_size == 512:
            x_comb = self.combiner_512(x1 + x2 + x3)
            
        # Combine features
        combined_features = []
        for idx, (f1, f2, f3) in enumerate(zip(features1, features2, features3)):
            combined = torch.cat([f1, f2, f3], dim=1)
            combined_features.append(self.feature_combiners[idx](combined))
            
        return self.future_decoder(x_comb, combined_features)



class DenoisingCNN_2048(nn.Module):
    def __init__(self, fine_layers, bottleneck_size):
        super(DenoisingCNN_2048, self).__init__()
        print("DenoisingCNN_2048")
        self.fine_layers = fine_layers
        self.bottleneck_size = bottleneck_size

        if self.fine_layers:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
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
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.future_decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
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
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

            self.future_decoder = nn.Sequential(
                nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
            )
        self.combiner_1024 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_512 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_256 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_128 = nn.Sequential(
            nn.Conv2d(1024, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_64 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_32 = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.combiner_16 = nn.Sequential(
            nn.Conv2d(1024, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )


    def forward(self, x):
        # x is concatenated input (B, 3*C, H, W)
        # Split into three frames
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        x3 = self.encoder(x3)
        if self.bottleneck_size == 16:
            x_comb = self.combiner_16(x1 + x2 + x3)
        elif self.bottleneck_size == 32:
            x_comb = self.combiner_32(x1 + x2 + x3)
        elif self.bottleneck_size == 64:
            x_comb = self.combiner_64(x1 + x2 + x3)
        elif self.bottleneck_size == 128:
            x_comb = self.combiner_128(x1 + x2 + x3)
        elif self.bottleneck_size == 256:
            x_comb = self.combiner_256(x1 + x2 + x3)
        elif self.bottleneck_size == 512:
            x_comb = self.combiner_512(x1 + x2 + x3)
        elif self.bottleneck_size == 1024:
            x_comb = self.combiner_1024(x1 + x2 + x3)
        x_decoded = self.future_decoder(x_comb)
        return x_decoded



class DenoisingCNN_2048_with_skips(DenoisingCNN_2048):
    def __init__(self, fine_layers, bottleneck_size):
        fine_layers = True  # fine_layers = True by default
        super(DenoisingCNN_2048_with_skips, self).__init__(fine_layers, bottleneck_size)
        print("DenoisingCNN_2048_with_skips")

        # Encoder blocks with internal skip connections
        self.enc1_conv = nn.Sequential(nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc1_pool = nn.MaxPool2d(2)
        self.enc2_conv = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc2_pool = nn.MaxPool2d(2)
        self.enc3_conv = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc3_pool = nn.MaxPool2d(2)
        self.enc4_conv = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc4_pool = nn.MaxPool2d(2)
        self.enc5_conv = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc5_pool = nn.MaxPool2d(2)
        self.enc6_conv = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc6_pool = nn.MaxPool2d(2)
        self.enc7_conv = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1), nn.ReLU(),)
        self.enc7_pool = nn.MaxPool2d(2)

        # Additional conv layers for encoder skip connections
        self.enc1_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, kernel_size=1)
        )
        self.enc2_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=1)
        )
        self.enc3_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=1)
        )
        self.enc4_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=1)
        )
        self.enc5_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(256, 256, kernel_size=1)
        )
        self.enc6_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(512, 512, kernel_size=1)
        )
        self.enc7_skip = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(1024, 1024, kernel_size=1)
        )

        # Decoder with skip connections
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=3, padding=1),  # 512 + 1024 = 1536
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=3, padding=1),  # 256 + 512 = 768
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.Conv2d(384, 128, kernel_size=3, padding=1),  # 128 + 256 = 384
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3, padding=1),  # 64 + 128 = 192
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec6 = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),  # 32 + 64 = 96
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec7 = nn.Sequential(
            nn.Conv2d(48, 16, kernel_size=3, padding=1),  # 16 + 32 = 48
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        )

        # Feature combiners (one for each feature level)
        self.feature_combiners = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c*3, c, kernel_size=3, padding=1),  # 3x channels because we concat 3 frames
                nn.ReLU(),
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.ReLU()
            ) for c in [16, 32, 64, 128, 256, 512, 1024]  # All channel sizes in encoder
        ])


    def encoder(self, x):
        """Encoder function that returns all intermediate features"""
        features = []
        # Encoder block 1
        x1 = self.enc1_conv(x)
        features.append(x1)  # Store pre-pool features
        x1_pool = self.enc1_pool(x1)
        x1 = x1_pool + self.enc1_skip(x1)
        # Encoder block 2
        x2 = self.enc2_conv(x1)
        features.append(x2)
        x2_pool = self.enc2_pool(x2)
        x2 = x2_pool + self.enc2_skip(x2)
        # Encoder block 3
        x3 = self.enc3_conv(x2)
        features.append(x3)
        x3_pool = self.enc3_pool(x3)
        x3 = x3_pool + self.enc3_skip(x3)
        # Encoder block 4
        x4 = self.enc4_conv(x3)
        features.append(x4)
        x4_pool = self.enc4_pool(x4)
        x4 = x4_pool + self.enc4_skip(x4)
        # Encoder block 5
        x5 = self.enc5_conv(x4)
        features.append(x5)
        x5_pool = self.enc5_pool(x5)
        x5 = x5_pool + self.enc5_skip(x5)
        # Encoder block 6
        x6 = self.enc6_conv(x5)
        features.append(x6)
        x6_pool = self.enc6_pool(x6)
        x6 = x6_pool + self.enc6_skip(x6)
        # Encoder block 7
        x7 = self.enc7_conv(x6)
        features.append(x7)
        x7_pool = self.enc7_pool(x7)
        x7 = x7_pool + self.enc7_skip(x7)
        return x7, features        

    def future_decoder(self, x, features):
        """Decoder function that uses stored features for skip connections"""
        features = features[::-1]
        # Decoder block 1
        d1 = self.dec1(x)  # 1024 -> 512
        d1 = torch.cat([d1, features[0]], dim=1)  # 512 + 1024 = 1536
        
        # Decoder block 2
        d2 = self.dec2(d1)  # Input: 1536
        d2 = torch.cat([d2, features[1]], dim=1)
        
        # Decoder block 3
        d3 = self.dec3(d2)
        d3 = torch.cat([d3, features[2]], dim=1)
        
        # Decoder block 4
        d4 = self.dec4(d3)
        d4 = torch.cat([d4, features[3]], dim=1)
        
        # Decoder block 5
        d5 = self.dec5(d4)
        d5 = torch.cat([d5, features[4]], dim=1)
        
        # Decoder block 6
        d6 = self.dec6(d5)
        d6 = torch.cat([d6, features[5]], dim=1)
        
        # Decoder block 7 (final block)
        out = self.dec7(d6)
        
        return out
    
    def forward(self, x): 
        # Split into three frames
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        
        # Get encoded representations and features
        x1, features1 = self.encoder(x1)
        x2, features2 = self.encoder(x2)
        x3, features3 = self.encoder(x3)
        
        # Combine encoded representations
        if self.bottleneck_size == 1024:
            x_comb = self.combiner_1024(x1 + x2 + x3)
        elif self.bottleneck_size == 512:
            x_comb = self.combiner_512(x1 + x2 + x3)
        elif self.bottleneck_size == 256:
            x_comb = self.combiner_256(x1 + x2 + x3)
        elif self.bottleneck_size == 128:
            x_comb = self.combiner_128(x1 + x2 + x3)
        elif self.bottleneck_size == 64:
            x_comb = self.combiner_64(x1 + x2 + x3)
        
        # Combine features using dedicated combiners
        combined_features = []
        for idx, (f1, f2, f3) in enumerate(zip(features1, features2, features3)):
            # Concatenate features from all frames
            combined = torch.cat([f1, f2, f3], dim=1)
            # Apply feature combiner
            combined_features.append(self.feature_combiners[idx](combined))
        
        # Decode with combined features
        return self.future_decoder(x_comb, combined_features)