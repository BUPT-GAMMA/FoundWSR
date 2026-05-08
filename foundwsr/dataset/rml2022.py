import numpy as np
import pandas as pd
import pickle
import os.path as osp
import torch
from torch.utils.data import Dataset
from scipy.signal import stft
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from . import register_dataset
from .base_dataset import BaseDataset

@register_dataset("RML2022")
class RML2022(BaseDataset):
    _processed_splits_cache = {}
    @classmethod
    def create(cls, dataset_path=None, *args, **kwargs):
        cls.signal_length = 128
        cls.split_list = ["train", "valid", "test"]
        cls.classes = ["8PSK", "AM-DSB", "AM-SSB", "BPSK", "CPFSK", "GFSK", "PAM4", "QAM16", "QAM64", "QPSK", "WBFM"]
        if dataset_path is None:
            dataset_path = osp.join(osp.dirname(osp.abspath(__file__)), "RML2022")
        f = open(osp.join(dataset_path, "RML22.01A"), "rb")
        cls.data = pickle.load(f, encoding="latinl")
        f.close()

    def __init__(self, split="train", test_size=0.2, val_size=0.25, target_class=None, *args, **kwargs):
        if not hasattr(RML2022, "data"):
            raise ValueError("The RML2022 dataset is not created, please use RML2022.create() to create instance.")
        if split not in self.split_list:
            raise ValueError(f"The split type {split} is not supported!")

        if target_class is not None:
            cache_key = (test_size, val_size, tuple(target_class))
        else:
            cache_key = (test_size, val_size, target_class)

        if cache_key not in RML2022._processed_splits_cache:
            if target_class is not None:
                indices = []
                common_class = []
                for label in target_class:
                    if label in self.classes:
                        common_class.append(label)

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
                maxSNR = 20

            self.minSNR = minSNR
            self.maxSNR = maxSNR
            self.SNR_list = range(minSNR, maxSNR + 1, 2)
            total = 0

            for item in self.data.items():
                (label, SNR), samples = item
                total = total + len(samples)
                if SNR < minSNR or SNR > maxSNR or label not in self.classes:
                    continue
                labels = np.full(len(samples), self.classes.index(label))
                SNR = np.full(len(samples), SNR)
                if "target_class" in kwargs:
                    for index, label in enumerate(labels):
                        if self.classes[label] in common_class:
                            indices.append(index)
                    samples = samples[indices]
                    SNR = SNR[indices]
                    labels = labels[indices]
                    label_to_new = {original_label: new_idx for new_idx, original_label in enumerate(common_class)}
                    labels = np.array([label_to_new[self.classes[x]] for x in labels])

                X, x, Y, y, SNR_tr, SNR_te = train_test_split(samples, labels, SNR, test_size=test_size,
                                                            random_state=233,
                                                            stratify=labels)
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
            RML2022._processed_splits_cache[cache_key] = splits
        else:
            cached_data = RML2022._processed_splits_cache[cache_key]
            self.dataset = cached_data["dataset"]
            self.classes = cached_data["classes"]
            self.SNR_list = cached_data["SNR_list"]
            self.minSNR = cached_data["minSNR"]
            self.maxSNR = cached_data["maxSNR"]
        print(np.unique(self.classes))
        import sys
        sys.exit()

    def __len__(self):
        return len(self.dataset[self.split_id][1])

    def __getitem__(self, idx):
        return torch.Tensor(self.dataset[self.split_id][0][idx]),\
            torch.tensor(self.dataset[self.split_id][1][idx], dtype=torch.long),\
            self.dataset[self.split_id][2][idx]

    def get_pretrain_data(self):
        return np.array(self.dataset[0][0]),\
                np.array(self.dataset[0][1]),\
                np.array(self.dataset[0][2])

