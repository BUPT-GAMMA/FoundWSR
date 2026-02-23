import os
import random
import numpy as np
import glob
import os.path as osp
import torch
from scipy.signal import stft
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from . import register_dataset
from .base_dataset import BaseDataset

@register_dataset("GNSS")
class GNSS(BaseDataset):
    _processed_splits_cache = {}
    @classmethod
    def create(cls, dataset_path=None, *args, **kwargs):
        cls.split_list = ["train", "valid", "test"]
        cls.classes = ["SingleAM", "SingleChirp", "NarrowBand", "NoJam", "SingleFM", "DME"]

        cls.minSNR = 0
        cls.maxSNR = 0
        cls.SNR_list = [0]
        cls.signal_length = 1024

        if dataset_path is None:
            dataset_path = osp.join(osp.dirname(osp.abspath(__file__)), "GNSS")

        if osp.exists(osp.join(dataset_path, "GNSS.npz")):
            print(osp.join(dataset_path, "GNSS.npz"))
            data = np.load(osp.join(dataset_path, "GNSS.npz"))
            cls.train_samples = data["train_samples"]
            cls.test_samples = data["test_samples"]
            cls.train_labels = data["train_labels"]
            cls.test_labels = data["test_labels"]
            data.close()
            print(cls.train_samples.shape)
            print(cls.test_samples.shape)
            print(cls.train_labels.shape)
            print(cls.test_labels.shape)
            import sys
            sys.exit()
        else:
            IQ_length = 1024
            train_files = glob.glob(osp.join(dataset_path, "Training/*/*.mat"))
            test_files = glob.glob(osp.join(dataset_path, "Testing/*/*.mat"))
            IQ_data_list = []
            label_list = []
            test_IQ_data_list = []
            test_label_list = []
            for file in train_files:
                infer_class = file.split("/")[-2]
                try:
                    infer_class = cls.classes.index(infer_class)
                except:
                    infer_class = cls.classes.index(infer_class)
                mat = loadmat(file)
                IQ = mat["GNSS_plus_Jammer_awgn"][0]
                n_segments = IQ.shape[0] // IQ_length
                for i in range (n_segments):
                    segment = IQ[i * IQ_length : (i + 1) * IQ_length]
                    IQ_segments = np.stack([segment.real, segment.imag], axis = 0)
                    IQ_data_list.extend(IQ_segments)
                    label_list.append(infer_class)

            for file in test_files:
                infer_class = file.split("/")[-2]
                try:
                    infer_class = cls.classes.index(infer_class)
                except:
                    infer_class = cls.classes.index(infer_class)
                mat = loadmat(file)
                IQ = mat["GNSS_plus_Jammer_awgn"][0]
                n_segments = IQ.shape[0] // IQ_length
                for i in range (n_segments):
                    segment = IQ[i * IQ_length : (i + 1) * IQ_length]
                    IQ_segments = np.stack([segment.real, segment.imag], axis = 0)
                    test_IQ_data_list.extend(IQ_segments)
                    test_label_list.append(infer_class)

            cls.train_samples = IQ_data_list
            cls.test_samples = test_IQ_data_list
            cls.train_labels = label_list
            cls.test_labels = test_label_list
            np.savez_compressed(osp.join(dataset_path, "GNSS.npz"),
                    train_samples=IQ_data_list,
                    train_labels=label_list,
                    test_samples=test_IQ_data_list,
                    test_labels=test_label_list)

    def __init__(self, split="train", test_size=0.2, val_size=0.25, target_class=None, *args, **kwargs):
        self.split_list = ["train", "valid", "test"]
        if not hasattr(GNSS, "train_samples"):
            raise ValueError("The Techrec dataset is not created, please use Techrec.create() to create instance.")
        if split not in self.split_list:
            raise ValueError(f"The split type {split} is not supported!")

        self.split_id = self.split_list.index(split)
        cache_key = (test_size, val_size, target_class)

        if cache_key not in GNSS._processed_splits_cache:
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

            self.minSNR = 0
            self.maxSNR = 0
            self.SNR_list = [0]

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
            GNSS._processed_splits_cache[cache_key] = splits
        else:
            cached_data = GNSS._processed_splits_cache[cache_key]
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
