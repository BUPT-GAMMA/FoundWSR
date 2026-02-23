import numpy as np
import torch
from abc import ABC
from torch.utils.data import Dataset
from scipy.signal import stft
from scipy.fft import fft

class BaseDataset(ABC, Dataset):
    @classmethod
    def create(cls, test_size, dataset_path):
        raise NotImplementedError("Datasets must implement the create method")
    def __init__(self, *args, **kwargs):
        super(BaseDataset, self).__init__()

class STFT:
    def __init__(self, fs=1.0, window='hann', nperseg=31, noverlap=30, nfft=128):
        self.fs = fs
        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft

    def forward(self, x):
        return stft(x[0,:], self.fs, self.window, self.nperseg, self.noverlap, self.nfft)

class FFT:
    def __init__(self, n_fft=None):
        super().__init__()
        self.n_fft = n_fft

    def forward(self, x):
        I = x[:, 0, :]
        Q = x[:, 1, :]
        s = torch.complex(I, Q)
        freq = fft(x, n=self.n_fft)
        re = freq.real
        im = freq.imag
        fft_data = np.concatenate([np.expand_dims(re, axis=1),
                                   np.expand_dims(im, axis=1)], axis=1)
        return fft_data