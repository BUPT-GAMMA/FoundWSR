import h5py
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

@register_dataset("RML2018")
class RML2018(BaseDataset):
    _processed_splits_cache = {}
    @classmethod
    def create(cls, dataset_path=None, *args, **kwargs):
        cls.split_list = ["train", "valid", "test"]
        cls.classes = ['OOK','4ASK','8ASK',
               'BPSK','QPSK','8PSK','16PSK','32PSK',
               '16APSK','32APSK','64APSK','128APSK',
               '16QAM','32QAM','64QAM','128QAM','256QAM',
               'AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC',
               'FM','GMSK','OQPSK']
        if dataset_path is None:
            dataset_path = osp.join(osp.dirname(osp.abspath(__file__)), "RML2018.01")
        with h5py.File(dataset_path + '/GOLD_XYZ_OSC.0001_1024.hdf5', 'r') as f:
            IQ_data = f['X'][:, :, :]
            class_label = f['Y'][:, :]
            SNR_label = f['Z'][:]
        cls.IQ_data = IQ_data.transpose(0, 2, 1)
        cls.class_labels = np.argmax(class_label, axis=1)
        cls.SNR_labels = SNR_label.squeeze()
        cls.signal_length = 1024

    def __init__(self, split="train", test_size=0.2, val_size=0.25, target_class=None, *args, **kwargs):
        if not hasattr(RML2018, "IQ_data"):
            raise ValueError("The RML2016 dataset is not created, please use RML2016.create() to create instance.")
        if split not in self.split_list:
            raise ValueError(f"The split type {split} is not supported!")

        self.split_id = self.split_list.index(split)
        cache_key = (test_size, val_size, target_class)

        if cache_key not in RML2018._processed_splits_cache:
            if target_class is not None:
                target_class = kwargs["target_class"]
                indices = []
                common_class = []
                for label in target_class:
                    if label in self.classes:
                        common_class.append(label)
                for index, label in enumerate(self.class_labels):
                    if self.classes[label] in common_class:
                        indices.append(index)
                IQ_data = self.IQ_data[indices]
                SNR_labels = self.SNR_labels[indices]
                labels = self.class_labels[indices]
                label_to_new = {original_label: new_idx for new_idx, original_label in enumerate(common_class)}
                class_labels = np.array([label_to_new[self.classes[x]] for x in labels])
                self.classes = common_class

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
                maxSNR = 30

            self.minSNR = minSNR
            self.maxSNR = maxSNR
            self.SNR_list = range(minSNR, maxSNR + 1, 2)

            X, x, Y, y, SNR_tr, SNR_te = train_test_split(IQ_data, class_labels, SNR_labels, test_size=test_size,
                                                        random_state=233,
                                                        stratify=class_labels)
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
            RML2018._processed_splits_cache[cache_key] = splits
        else:
            cached_data = RML2018._processed_splits_cache[cache_key]
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
        return np.array(self.dataset[0][0]), np.array(self.dataset[0][1]), np.array(self.dataset[0][2])

