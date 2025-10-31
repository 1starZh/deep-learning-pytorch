import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.modules):
    def __init__(self, input_dim, hidden_dim, laten_dim):
        super().__init__()
        self.encoder_dense = nn.Linear(input_dim, hidden_dim)
        self.mean_dense = nn.Linear(hidden_dim, laten_dim)
        self.log_var_dense = nn.Linear(hidden_dim, laten_dim)

    def forward(self, x):
        h = self.encoder_dense(x)
        h = F.relu(h)

        z_mean = self.mean_dense(h)
        z_log_var = self.log_var_dense(h)

        eps = torch.randn(z_mean.shape).to(z_mean.device)
        z = z_mean + torch.exp(z_log_var/2) * eps

        loss = -0.5 * torch.sum(1 + z_log_var - torch.square(z_mean) - torch.exp(z_log_var), dim=-1)

        return z, loss
    
class Decoder(nn.modules):
    def __init__(self, laten_dim, hidden_dim, origin_dim):
        super().__init__()
        self.decode_h = nn.Linear(laten_dim, hidden_dim)
        self.decode_x = nn.Linear(hidden_dim, origin_dim)

    def forward(self, x):
        h = self.decode_h(x)
        h = F.relu(h)
        x = self.decode_x(h)
        return x
    
class VAE(nn.modules):
    def __init__(self, origin_dim, hidden_dim, laten_dim):
        super().__init__()
        self.encoder = Encoder(origin_dim, hidden_dim, laten_dim)
        self.decoder = Decoder(laten_dim, hidden_dim, origin_dim)
        self.loss = nn.MSELoss(reduction=None)

    def forward(self, inputs):
        x, _ = torch.reshape(x, (-1, self.encoder.origin_dim))

        z, kl_loss = self.encoder(x)

        x_decoded = self.decoder(z)

        reconstruct_loss = torch.sum(self.loss(x_decoded, x), dim=-1)

        vae_loss = torch.mean(reconstruct_loss + kl_loss)

        return x_decoded, vae_loss

        @torch.no_grad()
        def generate(self, z):
            return self.decoder(z)