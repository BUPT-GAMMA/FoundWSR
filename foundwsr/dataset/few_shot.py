import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import stft
from ..utils.signal_data_augmentation import data_augmentation

class FewShotDataset(Dataset):
    def __init__(self, samples, labels, SNR, shot=25):
        SNR_list = np.unique(SNR)
        sample_data = []
        sample_label = []
        sample_snr = []
        unique_labels = np.unique(labels)
        for snr in SNR_list:
            mask = SNR == snr
            samples_filt = samples[mask]
            labels_filt = labels[mask]
            samples_of_snr = []
            labels_of_snr = []

            for l in unique_labels:
                idx = np.where(labels_filt == l)[0]
                chosen = np.random.choice(idx, size=shot, replace=False)
                samples_of_snr.append(samples_filt[chosen])
                labels_of_snr.append(labels_filt[chosen])

            samples_of_snr = np.concatenate(samples_of_snr)
            labels_of_snr = np.concatenate(labels_of_snr)
            sample_data.extend(samples_of_snr)
            sample_label.extend(labels_of_snr)
            snr_list = [snr] * len(samples_of_snr)
            sample_snr.extend(snr_list)

        self.samples = np.array(sample_data)
        self.labels = np.array(sample_label)
        self.SNR = np.array(sample_snr)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.Tensor(self.samples[idx]),\
                torch.tensor(self.labels[idx], dtype=torch.long),\
                self.SNR[idx]