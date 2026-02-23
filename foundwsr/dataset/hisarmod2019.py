import numpy as np
import pandas as pd
import os.path as osp
import torch
import h5py
from torch.utils.data import Dataset
from scipy.signal import stft
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from . import register_dataset
from .base_dataset import BaseDataset

@register_dataset("HisarMod2019")
class HisarMod2019(BaseDataset):
    _processed_splits_cache = {}
    @classmethod
    def create(cls, dataset_path=None, *args, **kwargs):
        cls.split_list = ["train", "valid", "test"]
        cls.classes = ['BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '64PSK', '4QAM', '8QAM', '16QAM', '32QAM', 
                   '64QAM', '128QAM', '256QAM', '2FSK', '4FSK', '8FSK', '16FSK', '4PAM', '8PAM', '16PAM', 'AM-DSB', 
                   'AM-DSB-SC', 'AM-USB', 'AM-LSB', 'FM', 'PM']

        if dataset_path is None:
            dataset_path = osp.join(osp.dirname(osp.abspath(__file__)), "HisarMod2019")

        with h5py.File(osp.join(dataset_path, 'HisarMod2019train.h5')) as h5file:
            cls.train = h5file['samples'][:]
            cls.train_label = h5file['labels'][:]
            cls.SNR_tr = h5file['snr'][:]
            h5file.close()

        with h5py.File(osp.join(dataset_path, 'HisarMod2019test.h5')) as h5file:
            cls.test = h5file['samples'][:]
            cls.test_label = h5file['labels'][:]
            cls.SNR_te = h5file['snr'][:]
            h5file.close()

    def __init__(self, split="train", test_size=0.2, val_size=0.25, target_class=None, *args, **kwargs):
        if not hasattr(HisarMod2019, "train"):
            raise ValueError("The HisarMod2019 dataset is not created, please use HisarMod2019.create() to create instance.")
        if split not in self.split_list:
            raise ValueError(f"The split type {split} is not supported!")

        self.split_id = self.split_list.index(split)
        cache_key = (test_size, val_size, target_class)

        if cache_key not in HisarMod2019._processed_splits_cache:
            if target_class is not None:
                target_class = kwargs["target_class"]
                indices = []
                common_class = []
                for label in target_class:
                    if label in self.classes:
                        common_class.append(label)
                for index, label in enumerate(self.train_label):
                    if self.classes[label] in common_class:
                        indices.append(index)
                train = self.train[indices]
                SNR_tr = self.SNR_tr[indices]
                labels = self.train_label[indices]
                label_to_new = {original_label: new_idx for new_idx, original_label in enumerate(common_class)}
                train_label = np.array([label_to_new[self.classes[x]] for x in labels])

                for index, label in enumerate(self.test_label):
                    if self.classes[label] in common_class:
                        indices.append(index)
                test = self.test[indices]
                SNR_te = self.SNR_te[indices]
                labels = self.test_label[indices]
                label_to_new = {original_label: new_idx for new_idx, original_label in enumerate(common_class)}
                test_label = np.array([label_to_new[self.classes[x]] for x in labels])

                self.classes = common_class

            else:
                train = self.train
                SNR_tr = self.SNR_tr
                train_label = self.train_label
                test = self.test
                SNR_te = self.SNR_te
                test_label = self.test_label

            train, val, train_label, val_label, SNR_tr, SNR_va = train_test_split(train, train_label, SNR_tr, test_size=test_size,
                                                                                random_state=233,
                                                                                stratify=list(zip(train_label,SNR_tr)))


            self.train_dataset = [[],[],[]]
            self.val_dataset = [[],[],[]]
            self.test_dataset = [[],[],[]]

            if "minSNR" in kwargs:
                minSNR = kwargs["minSNR"]
            else:
                minSNR = -20
            if "maxSNR" in kwargs:
                maxSNR = kwargs["maxSNR"]
            else:
                maxSNR = 18

            self.minSNR = minSNR
            self.maxSNR = maxSNR
            self.SNR_list = range(minSNR, maxSNR + 1, 2)

            self.train_dataset[0].extend(train)
            self.train_dataset[1].extend(train_label)
            self.train_dataset[2].extend(SNR_tr)
            self.val_dataset[0].extend(val)
            self.val_dataset[1].extend(val_label)
            self.val_dataset[2].extend(SNR_va)
            self.test_dataset[0].extend(test)
            self.test_dataset[1].extend(test_label)
            self.test_dataset[2].extend(SNR_te)
            self.dataset = [self.train_dataset, self.val_dataset, self.test_dataset]

            if "target_class" in kwargs:
                self.classes = common_class
            splits = {"dataset": self.dataset,
                      "classes": self.classes,
                      "SNR_list": self.SNR_list,
                      "minSNR": self.minSNR,
                      "maxSNR": self.maxSNR}
            HisarMod2019._processed_splits_cache[cache_key] = splits
        else:
            cached_data = HisarMod2019._processed_splits_cache[cache_key]
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

