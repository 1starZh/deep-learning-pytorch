import torch
from torch import nn
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.flatten_dim = 256 * 4 * 4

        self.fc_mean = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_dim, latent_dim)
    
    def forward(self, x):
        h = self.conv_layers(x) 
        h = h.view(h.size(0), -1)
        
        z_mean = self.fc_mean(h)
        z_log_var = self.fc_log_var(h)
        
        return z_mean, z_log_var
    
class ConvDecoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1] range for images
        )
    
    def forward(self, z):
        h = self.fc(z) 
        h = h.view(h.size(0), 256, 4, 4)  
        x_reconstructed = self.deconv_layers(h)
        return x_reconstructed

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)
    
    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)
        reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='sum') / x.size(0)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), 1), 0)
        beta = 1
        total_loss = reconstruction_loss + beta * kl_loss
        
        return x_reconstructed, total_loss, reconstruction_loss, kl_loss
    
    @torch.no_grad()
    def generate(self, num_samples=16, device='cpu'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        generated_images = self.decoder(z)
        return generated_images
    
    @torch.no_grad()
    def reconstruct(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed
