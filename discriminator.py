import torch.nn as nn
  
class Discriminator(nn.Module):
    def __init__(self, channels, img_size):
        super().__init__()
        self.channels = channels
        self.img_size = img_size

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]

            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.net = nn.Sequential(
            *discriminator_block(self.channels, 32, bn=False),
            *discriminator_block(32,64),
            *discriminator_block(64, 256),
            *discriminator_block(256, 512),
        )

        self.adv_layer = nn.Sequential(
            nn.Linear(512 * (self.img_size // 2 ** 4) ** 2, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        return self.adv_layer(x)