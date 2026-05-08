import os
import random
import numpy as np
import pandas as pd
import os.path as osp
import torch
from torch.utils.data import Dataset
from scipy.signal import stft
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from . import register_dataset
from .base_dataset import BaseDataset

def load_bin_file(file_path, segment_length=1024):
    # Read binary file as short integers
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.float32)
    # Reshape into (2, N) array for I and Q channels
    raw_data = raw_data.reshape(2, -1, order='F')  # F order to match MATLAB reshape
    
    # Convert to complex samples
    i_samples = raw_data[0]
    q_samples = raw_data[1]
    
    num_complete_segments = len(i_samples) // segment_length
    
    # Initialize list to store IQ segments
    iq_segments = []

    # Create segments
    for i in range(num_complete_segments):
        start_idx = i * segment_length
        end_idx = start_idx + segment_length
        
        segment = np.stack([
            i_samples[start_idx:end_idx],
            q_samples[start_idx:end_idx]
        ])
        
        iq_segments.append(segment)
    
    return iq_segments

def add_noise_awgn(x, snr):
    x = np.array(x)
    signal_power = np.mean(x ** 2, axis=(1, 2), keepdims=True)   # (B,1,1)
    noise_power = signal_power / (10 ** (snr / 10.0))          # (B,1,1)
    noise = np.random.normal(0, 1, x.shape).astype(x.dtype)
    noise *= np.sqrt(noise_power / 2)
    return (x + noise).tolist()

@register_dataset("Techrec")
class Techrec(BaseDataset):
    _processed_splits_cache = {}
    @classmethod
    def create(cls, dataset_path=None, *args, **kwargs):
        cls.split_list = ["train", "valid", "test"]
        cls.classes = ['lte', 'wf', 'dvbt']

        if "minSNR" in kwargs:
            minSNR = kwargs["minSNR"]
        else:
            minSNR = -15
        if "maxSNR" in kwargs:
            maxSNR = kwargs["maxSNR"]
        else:
            maxSNR = 20
        cls.minSNR = minSNR
        cls.maxSNR = maxSNR
        cls.SNR_list = range(minSNR, maxSNR + 1, 5)
        cls.signal_length = 1024

        if dataset_path is None:
            dataset_path = osp.join(osp.dirname(osp.abspath(__file__)), "Techrec")
        if osp.exists(osp.join(dataset_path, "Techrec.npz")):
            data = np.load(osp.join(dataset_path, "Techrec.npz"))
            cls.samples = data["samples"]
            cls.SNR = data["SNR"]
            cls.labels = data["labels"]
            data.close()
        else:
            dataset_info = []
            for root, dirs, files in os.walk(dataset_path):
                for file in files:
                    if file.endswith('.bin'):
                        parts = file.split('_')
                        if 'Msps' in parts[0]:
                            signal_type = ''.join([i for i in parts[0] if not i.isdigit() and i != 'M' and i != 's' and i != 'p'])
                            sampling_rate = int(''.join([i for i in parts[0] if i.isdigit()])) * 1e6
                        else:
                            signal_type = parts[0]
                            sampling_rate = 1e6
                        usrp = int(parts[1][1:])
                        location = root.split('/')[-1]
                        center_frequency = parts[3][1:-4] + 'MHz'
                        
                        dataset_info.append({
                            'signal_type': signal_type,
                            'sampling_rate': sampling_rate,
                            'usrp': usrp,
                            'location': location,
                            'center_frequency': center_frequency,
                            'file_path': os.path.join(root, file)
                        })

            IQ_data_list = []
            signal_type_label_list = []
            center_frequency_list = []
            location_list = []
            for sample in dataset_info:
                IQ_clips = load_bin_file(sample['file_path'], cls.signal_length)
                for IQ_clip in IQ_clips:
                    IQ_data_list.append(IQ_clip)
                    signal_type_label_list.append(cls.classes.index(sample['signal_type']))
                    center_frequency_list.append(sample['center_frequency'])
                    location_list.append(sample['location'])

            samples = []
            labels = []
            SNR = []
            num_snr = len(cls.SNR_list)
            N = len(IQ_data_list)
            k = N // num_snr
            combined = list(zip(IQ_data_list, signal_type_label_list))
            random.shuffle(combined)
            shuffled_IQ, shuffled_label = zip(*combined)
            shuffled_IQ = np.array(shuffled_IQ)
            shuffled_label = np.array(shuffled_label)

            for i, snr in enumerate(range(minSNR, maxSNR + 1, 5)):
                if (i + 1) == num_snr:
                    IQ_data = shuffled_IQ[i * k:]
                    label = shuffled_label[i * k:]
                else:
                    IQ_data = shuffled_IQ[i * k: (i + 1) * k]
                    label = shuffled_label[i * k: (i + 1) * k]
                SNR.extend([snr] * len(label))
                noised_sample = add_noise_awgn(IQ_data, snr)
                samples.extend(noised_sample)
                labels.extend(label)

            cls.samples = samples
            cls.SNR = SNR
            cls.labels = labels
            np.savez_compressed(osp.join(dataset_path, "Techrec.npz"),
                    samples=samples,
                    SNR=SNR,
                    labels=labels)

    def __init__(self, split="train", test_size=0.2, val_size=0.25, target_class=None, *args, **kwargs):
        self.split_list = ["train", "valid", "test"]
        if not hasattr(Techrec, "samples"):
            raise ValueError("The Techrec dataset is not created, please use Techrec.create() to create instance.")
        if split not in self.split_list:
            raise ValueError(f"The split type {split} is not supported!")

        if target_class is not None:
            cache_key = (test_size, val_size, tuple(target_class))
        else:
            cache_key = (test_size, val_size, target_class)

        if cache_key not in Techrec._processed_splits_cache:
            if target_class is not None:
                target_class = kwargs["target_class"]
                indices = []
                common_class = []
                for label in target_class:
                    if label in self.classes:
                        common_class.append(label)
                for index, label in enumerate(self.labels):
                    if self.classes[label] in common_class:
                        indices.append(index)
                samples = self.samples[indices]
                SNR = self.SNR[indices]
                labels = self.labels[indices]
                label_to_new = {original_label: new_idx for new_idx, original_label in enumerate(common_class)}
                labels = np.array([label_to_new[self.classes[x]] for x in labels])
                self.classes = common_class
            else:
                samples = self.samples
                SNR = self.SNR
                labels = self.labels

            if "minSNR" in kwargs:
                minSNR = kwargs["minSNR"]
            else:
                minSNR = -15
            if "maxSNR" in kwargs:
                maxSNR = kwargs["maxSNR"]
            else:
                maxSNR = 20

            self.minSNR = minSNR
            self.maxSNR = maxSNR
            self.SNR_list = range(minSNR, maxSNR + 1, 5)

            X, x, Y, y, SNR_tr, SNR_te = train_test_split(
                                        samples, 
                                        labels, 
                                        SNR, 
                                        test_size=test_size,
                                        random_state=233,
                                        stratify=labels)

            self.train_dataset = [[],[],[]]
            self.val_dataset = [[],[],[]]
            self.test_dataset = [[],[],[]]

            train, val, train_label, val_label, SNR_tr, SNR_va = train_test_split(X, Y, SNR_tr, test_size=val_size,
                                                                                random_state=233,
                                                                                stratify=Y)
            self.train_dataset[0].extend(train)
            self.train_dataset[1].extend(train_label)
            self.train_dataset[2].extend(SNR_tr)
            self.val_dataset[0].extend(val)
            self.val_dataset[1].extend(val_label)
            self.val_dataset[2].extend(SNR_va)
            self.test_dataset[0].extend(x)
            self.test_dataset[1].extend(y)
            self.test_dataset[2].extend(SNR_te)
            self.dataset = [self.train_dataset, self.val_dataset, self.test_dataset]

            if "target_class" in kwargs:
                self.classes = common_class
            splits = {"dataset": self.dataset,
                      "classes": self.classes,
                      "SNR_list": self.SNR_list,
                      "minSNR": self.minSNR,
                      "maxSNR": self.maxSNR}
            Techrec._processed_splits_cache[cache_key] = splits
        else:
            cached_data = Techrec._processed_splits_cache[cache_key]
            self.dataset = cached_data["dataset"]
            self.classes = cached_data["classes"]
            self.SNR_list = cached_data["SNR_list"]
            self.minSNR = cached_data["minSNR"]
            self.maxSNR = cached_data["maxSNR"]

    def __len__(self):
        return len(self.dataset[self.split_id][1])

    def __getitem__(self, idx):
        return torch.Tensor(self.dataset[self.split_id][0][idx]),\
            torch.tensor(self.dataset[self.split_id][1][idx], dtype=torch.long),\
            self.dataset[self.split_id][2][idx]

    def get_pretrain_data(self):
        return np.array(self.train_dataset[0]), np.array(self.train_dataset[1]), np.array(self.train_dataset[2])

