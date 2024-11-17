import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, channels):
        super().__init__()
        self.channels = channels
      
        self.init_size = (img_size // 4) // 2
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 256 * self.init_size ** 2)
        )

        self.l2 = nn.Sequential(
            nn.BatchNorm2d(256),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.ConvTranspose2d(64, self.channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.l1(x)
        x = x.view(x.shape[0], 256, self.init_size, self.init_size)
        x = self.l2(x)
        return x
    