import torch
import torch.nn as nn
import numpy as np

## **Variational Autoencoder**
# Outlines the VAE architecture (modifiable), including encoder, decoder, and probabilistic
# sampling of latent representation.

class VariationalAutoencoder(nn.Module):
    def __init__(self, img_channels=10, feature_dim=[128, 2, 2], latent_dim=128):
        super(VariationalAutoencoder, self).__init__()

        self.f_dim = feature_dim
        kernel_vae = 4
        stride_vae = 2

        # Initializing the convolutional layers and 2 full-connected layers for the encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=img_channels,
                      out_channels=128,
                      kernel_size=kernel_vae,
                      stride=stride_vae),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=kernel_vae,
                      stride=stride_vae),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=kernel_vae,
                      stride=stride_vae),
            nn.LeakyReLU())
        self.fc_mu = nn.Linear(np.prod(self.f_dim), latent_dim)
        self.fc_var = nn.Linear(np.prod(self.f_dim), latent_dim)

        # Initializing the fully-connected layer and convolutional layers for decoder
        self.dec_inp = nn.Linear(latent_dim, np.prod(self.f_dim))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=kernel_vae,
                               stride=stride_vae),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=128,
                               kernel_size=kernel_vae,
                               stride=stride_vae),
            nn.LeakyReLU(),
            # Final Layer
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=img_channels,
                               kernel_size=kernel_vae,
                               stride=stride_vae),
            nn.Sigmoid())

    def encode(self, x):
        # Input is fed into convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = self.encoder(x)
        x = x.view(-1, np.prod(self.f_dim))
        mu = self.fc_mu(x)
        logVar = self.fc_var(x)
        return mu, logVar

    def reparameterize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and samples the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        # z is fed back into a fully-connected layers and then into transpose convolutional layers
        # The generated output is the same size as the original input
        x = self.dec_inp(z)
        x = x.view(-1, self.f_dim[0], self.f_dim[1], self.f_dim[2])
        x = self.decoder(x)
        return x.squeeze()

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encode(x)
        z = self.reparameterize(mu, logVar)
        out = self.decode(z)
        return out, mu, logVar

# Print Architecture
print(VariationalAutoencoder())
