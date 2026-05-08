import torch
from torch.utils.data import Dataset
from scipy.signal import stft
from .signal_data_augmentation import data_augmentation

def add_noise(x, std):
    noise = torch.randn_like(x, device=x.device) * std
    return x + noise

def create_mask(input_data, mask_ratio=0.3):
    """
    input_data: Tensor of shape [seq_length, input_dim]
    mask_ratio: The ratio of the sequence to mask (0 to 1)
    
    Returns:
        masked_input: The input with masked positions (shape [seq_length, input_dim])
        mask_matrix: Mask matrix with 1 for unmasked and 0 for masked positions (shape [seq_length])
        pre_label: The last element in the sequence (shape [input_dim])
    """
    seq, input_dim = input_data.shape  # For a single sample, shape is [seq_length, input_dim]
    mask_matrix = torch.ones(seq).to(input_data.device)
    num_masked = int(seq * mask_ratio)
    mask_indices = torch.randperm(seq)[:num_masked]
    mask_matrix[mask_indices] = 0
    mask_matrix[-1] = 0
    masked_input = torch.multiply(input_data, mask_matrix.unsqueeze(-1))  # Broadcasting to match the shape
    pre_label = input_data[-1, :]

    return masked_input, mask_matrix, pre_label

class MaskedDataset(Dataset):
    def __init__(self, samples, labels, SNR, noise_std, mask_ratio):
        self.samples = samples
        self.labels = labels
        self.SNR = SNR
        self.std = noise_std
        self.mask_ratio = mask_ratio

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.tensor(self.samples[idx], dtype=torch.float).transpose(0, 1)
        label = torch.tensor(self.labels, dtype=torch.long)[idx]
        snr = torch.tensor(self.SNR, dtype=torch.long)[idx]
        noise_data = add_noise(sample, self.std)
        masked_data, mask, pre_label = create_mask(noise_data, self.mask_ratio)

        return sample, label, snr, masked_data, mask, pre_label

class MoCoPretrainDataset(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = samples
        self.SNR = SNR
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        _, _, stp = stft(self.samples[idx][0,:], 1.0, 'blackman', 31, 30, 128)
        augmented_samples = data_augmentation(self.samples[idx])
        _, _, stp_augmented = stft(augmented_samples[0,:], 1.0, 'blackman', 31, 30, 128)
        
        return self.samples[idx], augmented_samples, stp, stp_augmented

class PretrainDataset(Dataset):
    def __init__(self, samples, labels, SNR):
        self.samples = samples
        self.SNR = SNR
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.SNR[idx]