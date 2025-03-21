import torch
from torch import nn

class DenoisingCNN_512_prev_frames_5(nn.Module):
    def __init__(self):
        super(DenoisingCNN_512_prev_frames_5, self).__init__()
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
        )
        self.future_decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2),
        )
        
        self.combiner = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        frame1, frame2, frame3, frame4, frame5, frame6 = torch.chunk(x, 6, dim=1)
        x1 = self.encoder(frame1)
        x2 = self.encoder(frame2)
        x3 = self.encoder(frame3)
        x4 = self.encoder(frame4)
        x5 = self.encoder(frame5)
        x6 = self.encoder(frame6)
        x = self.combiner(x1 + x2 + x3 + x4 + x5 + x6)
        x = self.future_decoder(x)
        return x


class DenoisingCNN_1024_prev_frames_5(nn.Module):
    def __init__(self):
        super(DenoisingCNN_1024_prev_frames_5, self).__init__()
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

        self.combiner = nn.Sequential(
            nn.Conv2d(512, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        frame1, frame2, frame3, frame4, frame5, frame6 = torch.chunk(x, 6, dim=1)
        x1 = self.encoder(frame1)
        x2 = self.encoder(frame2)
        x3 = self.encoder(frame3)
        x4 = self.encoder(frame4)
        x5 = self.encoder(frame5)
        x6 = self.encoder(frame6)
        x = self.combiner(x1 + x2 + x3 + x4 + x5 + x6)
        x = self.future_decoder(x)
        return x
