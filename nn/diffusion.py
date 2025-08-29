import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


def sigmoid_schedule(T, beta_start=0.0001, beta_end=0.01, tau=5.0):
    steps = torch.linspace(-tau, tau, T)
    sig = torch.sigmoid(steps)
    sig = (sig - sig.min()) / (sig.max() - sig.min()) 
    betas = beta_start + (beta_end - beta_start) * sig
    return betas


def cosine_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize to start at 1

    alpha_t = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alpha_t

    return torch.clamp(betas, 0.0001, 0.01)


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer('betas', sigmoid_schedule(T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, c):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

        loss = F.mse_loss(self.model(x_t, t, c), noise)
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, T, iT, eta):
        super().__init__()

        self.model = model
        self.T = T
        self.iT = iT
        self.eta = eta

        self.register_buffer('betas', sigmoid_schedule(T).double())
        alphas = 1.0 - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_bar', alphas_bar)

        self.ddim_timesteps = np.linspace(0, T-1, iT, dtype=int)

    def predict_x0(self, x_t, t, eps):
        alphas_bar_t = extract(self.alphas_bar, t, x_t.shape)
        return (x_t - torch.sqrt(1 - alphas_bar_t) * eps) / torch.sqrt(alphas_bar_t)

    def ddim_step(self, x_t, t, t_prev, x_c):
        eps = self.model(x_t, t, x_c)

        x0_pred = self.predict_x0(x_t, t, eps)

        a_t = extract(self.alphas_bar, t, x_t.shape)
        a_prev = extract(self.alphas_bar, t_prev, x_t.shape)

        sigma = self.eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
        noise = torch.randn_like(x_t) if sigma > 0 else 0

        mean = torch.sqrt(a_prev) * x0_pred + torch.sqrt(1 - a_prev - sigma**2) * eps
        x_prev = mean + sigma * noise
        return x_prev

    def forward(self, x_T, x_c):
        x_t = x_T
        for i in reversed(range(len(self.ddim_timesteps))):
            print(i)
            t = x_t.new_full((x_t.shape[0],), self.ddim_timesteps[i], dtype=torch.long)
            if i == 0:
                t_prev = t
            else:
                t_prev = x_t.new_full((x_t.shape[0],), self.ddim_timesteps[i-1], dtype=torch.long)

            x_t = self.ddim_step(x_t, t, t_prev, x_c)
        return x_t