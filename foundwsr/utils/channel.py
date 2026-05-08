import torch
import random
from torch.distributions import Gamma

def apply_rician(Y0, w):
    B, L = Y0.shape
    # K = random.randint(1, 6)
    los_power = (1 - w)**2      # 0.8
    scatter_power = 1 - los_power  # 0.2
    scatter_power = scatter_power.view(B, 1)
    los = torch.sqrt(torch.tensor(los_power, device=Y0.device)).view(B, 1)
    scatter = (torch.randn(B, L, device=Y0.device) + 
               1j * torch.randn(B, L, device=Y0.device)) * torch.sqrt(torch.tensor(scatter_power) / 2)

    H = los + scatter
    return Y0 * H

def apply_cfo(Y0, w):
    B, L = Y0.shape
    sample_rate = 200e3
    max_cfo_hz = w * 500
    cfo_hz = (torch.rand(B, device=Y0.device) * 2 - 1) * max_cfo_hz
    shift = (cfo_hz / sample_rate * L).int()

    Y_shifted = torch.zeros_like(Y0)
    for b in range(B):
        Y_shifted[b] = torch.roll(Y0[b], shifts=shift[b].item(), dims=-1)
    return Y_shifted

def apply_sro(Y0, w):
    B, L = Y0.shape
    sample_rate = 200e3
    max_sro_hz = w * 1000  # 1% of 100 kHz bandwidth
    sro_hz = torch.randn(B, device=Y0.device) * (max_sro_hz)
    shift = (sro_hz / sample_rate * L / 2).int()

    Y_shifted = torch.zeros_like(Y0)
    for b in range(B):
        Y_shifted[b] = torch.roll(Y0[b], shifts=shift[b].item(), dims=-1)
    return Y_shifted

def apply_awgn(Y0, w):
    B, L = Y0.shape

    snr_db = 40.0 * (0.5 - w)  # [B]
    snr_linear = 10 ** (snr_db / 10)  # [B]

    # Signal power per sample (assume normalized input)
    signal_power = torch.mean(Y0.abs() ** 2, dim=-1, keepdim=True)  # [B, 1]

    noise_power = signal_power / snr_linear.unsqueeze(-1)  # [B, 1]
    sigma = torch.sqrt(noise_power / 2.0)  # per real/imag

    noise_real = torch.randn(B, L, device=Y0.device) * sigma
    noise_imag = torch.randn(B, L, device=Y0.device) * sigma
    Y_noisy = Y0 + (noise_real + 1j * noise_imag)

    return Y_noisy

def apply_sto(Y0, w):
    B, L = Y0.shape

    # Max fractional delay: 0.5 symbol → L/2 samples
    max_delay = L / 2.0
    tau = w * max_delay  # [B]

    # Time domain
    x_time = torch.fft.ifft(Y0, dim=-1)  # [B, L]

    # Create sinc kernel (simplified: use linear interp for efficiency)
    n = torch.arange(L, device=Y0.device).float()  # [L]
    tau = tau.unsqueeze(-1)  # [B, 1]

    n_shifted = n - tau  # [B, L]
    n_shifted = torch.clamp(n_shifted, 0, L - 1)

    idx_low = n_shifted.long()
    idx_high = torch.clamp(idx_low + 1, max=L - 1)
    w_high = n_shifted - idx_low.float()
    w_low = 1 - w_high

    x_low = x_time.gather(1, idx_low)
    x_high = x_time.gather(1, idx_high)
    x_delayed = w_low * x_low + w_high * x_high

    Y_out = torch.fft.fft(x_delayed, dim=-1)
    return Y_out

def apply_rayleigh(Y0, w):
    B, L = Y0.shape

    # When w=0 → h = 1 (no fading)
    # When w=1 → h ~ CN(0,1) (full Rayleigh)
    # Interpolate between identity and Rayleigh channel
    h_rayleigh = (torch.randn(B, L, device=Y0.device) + 
                  1j * torch.randn(B, L, device=Y0.device)) / torch.sqrt(torch.tensor(2.0))

    h = (1 - w).unsqueeze(-1) + w.unsqueeze(-1) * h_rayleigh  # [B, L]
    Y_out = h * Y0
    return Y_out

def apply_nakagami(Y0, w, m_min=0.5, m_max=20.0):
    B, L = Y0.shape
    device = Y0.device

    # Map w ∈ [0,1] to m ∈ [m_max, m_min] (monotonic)
    m = m_min + (m_max - m_min) * (1 - w)  # [B]
    m = m.unsqueeze(-1)  # [B, 1]

    # |h|^2 ~ Gamma(m, 1/m) → E[|h|^2] = 1

    gamma_dist = Gamma(concentration=m, rate=m)  # rate = 1/scale
    power = gamma_dist.rsample()  # [B, 1], positive, E=1
    magnitude = torch.sqrt(power)  # [B, 1]

    # Uniform phase in [0, 2π)
    phase = torch.rand(B, L, device=device) * 2 * torch.pi  # [B, L]
    h = magnitude * torch.exp(1j * phase)  # [B, L]
    Y_out = h * Y0
    return Y_out

def apply_multipath(Y0, w, max_taps=5):
    """
    Frequency-selective fading via random FIR filter.
    w=0: no multipath (h = delta); w=1: full multipath with max_taps.
    """
    B, L = Y0.shape

    # Generate random complex FIR taps (exponential power delay profile)
    # Power decays as exp(-k / tau), tau ~ 1~3 taps
    k = torch.arange(max_taps, device=Y0.device).float()  # [T]
    tau = 2.0  # avg delay spread in taps
    power_profile = torch.exp(-k / tau)  # [T]
    power_profile = power_profile / power_profile.sum()  # normalize

    # Scale total power by w: w=0 → all power in first tap; w=1 → full profile
    base_power = torch.zeros_like(power_profile)
    base_power[0] = 1.0
    final_power = (1 - w).unsqueeze(-1) * base_power + w.unsqueeze(-1) * power_profile  # [B, T]

    # Generate complex taps with given power
    real_taps = torch.randn(B, max_taps, device=Y0.device) * torch.sqrt(final_power / 2)
    imag_taps = torch.randn(B, max_taps, device=Y0.device) * torch.sqrt(final_power / 2)
    h_taps = real_taps + 1j * imag_taps  # [B, T]

    # Convolve in time domain (circular convolution via FFT for efficiency)
    H_freq = torch.fft.fft(h_taps, n=L, dim=-1)  # [B, L]
    X_freq = Y0  # already in freq domain
    Y_out = X_freq * H_freq  # multiplication in freq = circular conv in time

    return Y_out
