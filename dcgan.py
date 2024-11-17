import torch
import torch.nn as nn
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
from generator import Generator
from discriminator import Discriminator
from weights import weights_init_normal

save_path = "DCGAN"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# Hyperparameters --------------------------------------------------------
img_size = 64
latent_dim = 100
channels = 3
batch_size = 64
lr = 0.0001
b1 = 0.5
b2 = 0.999
epochs = 200

# Loading dataset --------------------------------------------------------
dataset_path = "anime"
dataset = datasets.ImageFolder(root=dataset_path, 
    transform=transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)
dataset = torch.utils.data.Subset(dataset, range(10000))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create models & optimisers ----------------------------------------------
loss = nn.BCEWithLogitsLoss()
generator = Generator(img_size, latent_dim, channels)
discriminator = Discriminator(channels, img_size)

if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()
    loss.cuda()

generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimiser_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimiser_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor()
scaler = torch.amp.GradScaler("cuda") # Mixed precision training

# Training loop ------------------------------------------------------------
for epoch in range(epochs + 1):
    for i, (imgs,_) in enumerate(dataloader):
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        # valid = Variable(Tensor(imgs.shape[0], 1).fill_(0.9), requires_grad=False) # Use for label smoothing
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        real_imgs = Variable(imgs.type(Tensor))

        optimiser_g.zero_grad()
        x = Variable(Tensor(torch.randn(imgs.shape[0], latent_dim).cuda()))
        gen_imgs = generator(x)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            g_loss = loss(discriminator(gen_imgs), valid)

        scaler.scale(g_loss).backward()
        scaler.step(optimiser_g)
        scaler.update()

        optimiser_d.zero_grad()

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            real_loss = loss(discriminator(real_imgs), valid)
            fake_loss = loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

        scaler.scale(d_loss).backward()
        scaler.step(optimiser_d)
        scaler.update()

    print(f"Epoch [{epoch}/ {epochs}]: D loss {d_loss.item()}, G loss: {g_loss.item()}")

    if epoch % 5 == 0:
        save_image(gen_imgs.data, save_path + "/Epoch-%d.png" % epoch, normalize=True)
        print("Saved images.")

# Create batches of images with fully trained model --------------------------------
for i in range(10):
    x = Variable(Tensor(torch.randn(imgs.shape[0], latent_dim).cuda()))
    gen_imgs = generator(x)
    save_image(gen_imgs.data, save_path + "/DCGAN-%d.png" % i, normalize=True)

print("Finished training!")