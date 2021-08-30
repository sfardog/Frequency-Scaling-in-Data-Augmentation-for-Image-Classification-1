import torch
import torch.nn as nn

class NetworkA1(nn.Module) :
    def __init__(self, channels=2):
        super(NetworkA1, self).__init__()

        self.neta1 = nn.Sequential(
            nn.Conv2d(channels, out_channels=16, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(7, 7), padding=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(5, 5), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, channels//2, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, image1, image2):
        inp = torch.cat([image1, image2], dim=1)
        out = self.neta1(inp)

        return out