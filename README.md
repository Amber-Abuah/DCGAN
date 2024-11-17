## DCGAN
Deep Convolutional Generative Adversarial Network (DCGAN) inspired by [PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py) and [DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).

Trained to generate 64x64 anime faces from [20,000 training images](https://huggingface.co/datasets/huggan/anime-faces).

Mixed precision training to speed up training times.

### Generator Architecture
- Linear (Latent Dim -> Img size / 8 ** 2)
- BatchNorm2d
- ConvTranspose2d (256 -> 128)
- BatchNorm2d
- Leaky ReLU
- ConvTranspose2d (128 -> 64)
- BatchNorm2d
- Leaky ReLU
- ConvTranspose2d (64 -> Num Colour Channels)
- Tanh

### Discriminator Architecture
- Conv2d(3 -> 32)
- Leaky ReLU
- Dropout(0.25)
- Conv2d (32 -> 64)
- Leaky ReLU
- Dropout(0.25)
- BatchNorm2d
- Conv2d (64 -> 256)
- Leaky ReLU
- Dropout(0.25)
- BatchNorm2d
- Conv2d (256 -> 512)
- Leaky ReLU
- Dropout(0.25)
- BatchNorm2d
- Linear (512 -> 1)
- Sigmoid (unapplied in example as loss function is BCEWithLogitsLoss)