import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def extract(a, t, x):
    b, *_ = t.shape
    out = a[t]
    return out.reshape(b, *((1,) * (len(x.shape) - 1))) #[b, 1, 1, 1]

class SimpleDiffusionModel(nn.Module):
    def __init__(self, model, img_size, img_channels, num_steps=1000, device='cuda'):
        super().__init__()
        self.model = model
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_steps = num_steps
        self.device = device
        
        self.betas = torch.linspace(0.0001, 0.02, self.num_steps, dtype=torch.float32, device=device)
        self.alphas = (1.0 - self.betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.reciprocal_sqrt_alphas = torch.sqrt(1.0 / self.alphas)
        self.remove_noise_coeff = self.betas / self.sqrt_one_minus_alphas_cumprod
        self.sigma = torch.sqrt(self.betas)
        
    def perturb(self, x, t, noise):
        return extract(self.sqrt_alphas_cumprod, t, x) * x + extract(self.sqrt_one_minus_alphas_cumprod, t, x) * noise
    
    @torch.no_grad()
    def remove_noise(self, x, t):
        return (x - extract(self.remove_noise_coeff, t, x) * self.model(x, t)) * extract(self.reciprocal_sqrt_alphas, t, x)
    
    @torch.no_grad()
    def sample(self, batch):
        x = torch.randn(batch, self.img_channels, *self.img_size, device=self.device)
        
        for i in range(self.num_steps - 1, -1, -1):
            t_batch = torch.tensor([i], device=self.device).repeat(batch)
            x = self.remove_noise(x, t_batch)
            if i > 0:
                x += torch.randn_like(x) * extract(self.sigma, t_batch, x.shape)
                
        return x
    
    def get_loss(self, x0, t):
        noise = torch.randn_like(x0)
        xt = self.perturb(x0, t, noise)
        predicted_noise = self.model(xt, t)
        return F.mse_loss(noise, predicted_noise)
    
    def forward(self, x):
        b, c, h, w = x.shape
        device = x.device
        t = torch.randint(0, self.num_steps, (b,), device=device).long()
        return self.get_loss(x, t)
    
    @torch.no_grad()
    def sample_ddim(self, batch, ddim_steps=50, eta=0.0):
        t_ddim = np.linspace(0, self.num_steps - 1, ddim_steps, dtype=int)
        alphas_cumprod = self.alphas_cumprod
        ddim_alphas = alphas_cumprod[t_ddim]
        ddim_sqrt_one_minus_alphas = torch.sqrt(1.0 - ddim_alphas)
        ddim_alphas_prev = torch.cat([alphas_cumprod[0:1], ddim_alphas[:-1]])
        ddim_sigmas = eta * torch.sqrt(
            (1.0 - ddim_alphas_prev) / (1.0 - ddim_alphas) * 
            (1.0 - ddim_alphas / ddim_alphas_prev)
        )
        
        x = torch.randn(batch, self.img_channels, *self.img_size, device=self.device)
        for i in range(ddim_steps - 1, -1, -1):
            t = torch.tensor([t_ddim[i]], device=self.device).repeat(batch)
            alpha = ddim_alphas[i]
            alpha_prev = ddim_alphas_prev[i]
            sqrt_one_minus_alpha = ddim_sqrt_one_minus_alphas[i]
            
            pred_noise = self.model(x, t)
            x0_pred = (x - sqrt_one_minus_alpha * pred_noise) / torch.sqrt(alpha)
            
            if i > 0:
                sigma = ddim_sigmas[i]
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev - sigma**2) * pred_noise + sigma * noise
            else:
                x = x0_pred
        return x