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

@register_dataset("RML2016")
class RML2016(BaseDataset):
    _processed_splits_cache = {}
    @classmethod
    def create(cls, dataset_path=None, dataset_name="RML2016.10a", *args, **kwargs):
        cls.dataset_name = dataset_name
        cls.split_list = ["train", "valid", "test"]
        if dataset_path is None:
            dataset_path = osp.join(osp.dirname(osp.abspath(__file__)), dataset_name)
        if dataset_name == "RML2016.10b":
            cls.data = pd.read_pickle(osp.join(dataset_path, 'RML2016.10b.dat'))
            cls.classes = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'WBFM']
            cls.signal_length = 128
        elif dataset_name == "RML2016.04c":
            cls.data = pd.read_pickle(osp.join(dataset_path, '2016.04C.multisnr.pkl'))
            cls.classes = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'AM-SSB', 'WBFM']
            cls.signal_length = 128
        else:
            cls.data = pd.read_pickle(osp.join(dataset_path, 'RML2016.10a_dict.pkl'))
            cls.classes = ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'AM-SSB', 'WBFM']
            cls.signal_length = 128

    def __init__(self, split="train", test_size=0.2, val_size=0.25, target_class=None, *args, **kwargs):
        if not hasattr(RML2016, "data"):
            raise ValueError("The RML2016 dataset is not created, please use RML2016.create() to create instance.")
        if split not in self.split_list:
            raise ValueError(f"The split type {split} is not supported!")

        self.split_id = self.split_list.index(split)
        if target_class is not None:
            cache_key = (test_size, val_size, tuple(target_class))
        else:
            cache_key = (test_size, val_size, target_class)

        if cache_key not in RML2016._processed_splits_cache:
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

            for item in self.data.items():
                (label, SNR), samples = item
                if SNR < minSNR or SNR > maxSNR or label not in self.classes:
                    continue
                label_id = self.classes.index(label)
                SNR = np.full(len(samples), SNR)
                labels = np.full(len(samples), label_id)

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

            if target_class is not None:
                common_class = []
                for label in target_class:
                    if label in self.classes:
                        common_class.append(label)
                label_to_new = {original_label: new_idx for new_idx, original_label in enumerate(common_class)}
                for idx, dataset in enumerate(self.dataset):
                    data = dataset[0]
                    labels = dataset[1]
                    SNR = dataset[2]
                    indices = [index for index, label in enumerate(labels) if self.classes[label] in common_class]
                    data = [data[i] for i in indices]
                    labels = [label_to_new[self.classes[labels[i]]] for i in indices]
                    SNR = [SNR[i] for i in indices]
                    self.dataset[idx] = [data, labels, SNR]
                self.classes = common_class

            splits = {"dataset": self.dataset,
                      "classes": self.classes,
                      "SNR_list": self.SNR_list,
                      "minSNR": self.minSNR,
                      "maxSNR": self.maxSNR}
            RML2016._processed_splits_cache[cache_key] = splits
        else:
            cached_data = RML2016._processed_splits_cache[cache_key]
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
        return np.array(self.dataset[0][0]),\
                np.array(self.dataset[0][1]),\
                np.array(self.dataset[0][2])

