import torch
import copy

class Diffusion:
    def __init__(self, max_step, min_noise, max_noise, device):
        self.timesteps = max_step
        self.betas = torch.linspace(min_noise, max_noise, max_step, dtype=torch.float64).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    @torch.compile
    def q_sample(self, x0, t, noise=None):
        t = t.to("cpu")
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod[t])
        for _ in range(x0.dim() - 1):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
        x_noise = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise

        return x_noise.to(torch.float32), noise

    @torch.compile
    def p_sample(self, model, x, t):
        t = torch.full((x.size(0),), t, device=x.device, dtype=torch.long)
        noise_pred = model(x, t)
        for _ in range(x.dim() - 1):
            alpha = self.alphas[t].unsqueeze(-1)
        for _ in range(x.dim() - 1):
            alpha_cumprod = self.alpha_cumprod[t].unsqueeze(-1)
        one_minus_alpha_cumprod = 1.0 - alpha_cumprod
        sqrt_one_minus_alpha_cumprod = torch.sqrt(one_minus_alpha_cumprod)
        posterior_mean = (x - (1 - alpha) / sqrt_one_minus_alpha_cumprod * noise_pred) / torch.sqrt(alpha)
        if t[0] > 0:
            noise = torch.randn_like(x)
            return posterior_mean + torch.sqrt(1 - alpha) * noise
        return posterior_mean

class Time_Freq_Diffusion:
    def __init__(self, max_step, min_noise, max_noise, ratio, device):
        self.timesteps = max_step
        self.betas = torch.linspace(min_noise, max_noise, max_step, dtype=torch.float64).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.beta_cumprod = ratio * self.sqrt_one_minus_alpha_cumprod
        self.gamma_cumprod = (1 - ratio) * self.sqrt_one_minus_alpha_cumprod

    @torch.compile
    def q_sample(self, x0, t, noise=None):
        t = t.to("cpu")
        if noise is None:
            epsilon = torch.randn_like(x0)
            eta = torch.randn_like(x0)

        sqrt_alpha_cumprod = self.sqrt_alpha_cumprod[t]
        beta_cumprod = self.beta_cumprod[t]
        gamma_cumprod = self.gamma_cumprod[t]
        for _ in range(x0.dim() - 1):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            beta_cumprod = beta_cumprod.unsqueeze(-1)
            gamma_cumprod = gamma_cumprod.unsqueeze(-1)
        x_noise = sqrt_alpha_cumprod * x0 + beta_cumprod * epsilon + gamma_cumprod * eta

        return x_noise.to(torch.float32), epsilon, eta, (gamma_cumprod / beta_cumprod) ** 2