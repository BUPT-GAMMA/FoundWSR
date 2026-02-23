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

@register_dataset("ICARUS")
class ICARUS(BaseDataset):
    @classmethod
    def create(cls, test_size=0.2, dataset_path=None, type="tone", *args, **kwargs):
        # type, only support "tone", "chirp", "pulse"
        # self.label = torch.tensor(labels, dtype=torch.long)

        if dataset_path is None:
            dataset_path = osp.join(osp.join(osp.dirname(osp.abspath(__file__)), "ICARUS"), "ICARUS.npz")

        data = np.load(dataset_path)
        samples = data["samples"]
        labels = data["labels"]
        SIR = data["SIR"]
        cls.SIR_list = np.unique(SIR)
        
        cls.signal_length = 1024
        X, x, Y, y, SIR_tr, SIR_te = train_test_split(
                                    samples, 
                                    labels, 
                                    SIR, 
                                    test_size=test_size,
                                    random_state=233,
                                    stratify=labels)

        cls.train_dataset = [[],[],[]]
        cls.val_dataset = [[],[],[]]
        cls.test_dataset = [[],[],[]]

        train, val, train_label, val_label, SIR_tr, SIR_va = train_test_split(X, Y, SIR_tr, test_size=0.25,
                                                                            random_state=233,
                                                                            stratify=Y)

        cls.train_dataset[0].extend(train)
        cls.train_dataset[1].extend(train_label)
        cls.train_dataset[2].extend(SIR_tr)
        cls.val_dataset[0].extend(val)
        cls.val_dataset[1].extend(val_label)
        cls.val_dataset[2].extend(SIR_va)
        cls.test_dataset[0].extend(x)
        cls.test_dataset[1].extend(y)
        cls.test_dataset[2].extend(SIR_te)
        cls.dataset = [cls.train_dataset, cls.val_dataset, cls.test_dataset]

    def __init__(self, split="train"):
        split_list = ["train", "valid", "test"]
        if not hasattr(ICARUS, "train_dataset"):
            raise ValueError("The RML2016 dataset is not created, please use RML2016.create() to create instance.")
        if split not in split_list:
            raise ValueError(f"The split type {split} is not supported!")
        
        self.split_id = split_list.index(split)
        self.split = split

    def __len__(self):
        if self.split == "train":
            return len(self.train_dataset[0])
        elif self.split == "valid":
            return len(self.val_dataset[0])
        elif self.split == "test":
            return len(self.test_dataset[0])

    def __getitem__(self, idx):
        return torch.Tensor(self.dataset[self.split_id][0][idx], dtype=torch.float),\
            torch.tensor(self.dataset[self.split_id][1][idx], dtype=torch.long),\
            self.dataset[self.split_id][2][idx]

    @property
    def get_pretrain_data(self):
        return np.array(self.train_dataset[0]), np.array(self.train_dataset[1]), np.array(self.train_dataset[2])
