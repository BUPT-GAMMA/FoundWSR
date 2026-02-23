from torch.utils.data import Dataset
from scipy.signal import stft
from ..utils.signal_data_augmentation import data_augmentation

class PretrainDataset(Dataset):
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
        # try:
        #     IQ_original = torch.Tensor(self.samples[idx])
        #     IQ_agumented = torch.Tensor(agumented_samples)

        #     stp_original = torch.Tensor(np.expand_dims(stp[:32,:], 0))
        #     stp_agumented = torch.Tensor(np.expand_dims(stp_agumented[:32,:], 0))
        # except:
        #     print(f"Error: {self.samples[idx]}")
        #     return None
        
        return self.samples[idx], augmented_samples, stp, stp_augmented