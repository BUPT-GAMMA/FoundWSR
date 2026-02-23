import torch.nn.functional as F
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..dataset.masked import MaskedDataset

@register_task("masked_reconstruction")
class MaskedReconstruction(BaseTask):
    def __init__(self, args):
        super(MaskedReconstruction, self).__init__()
        self.args = args
        if hasattr(self.args, "noise_std"):
            self.noise_std = args.noise_std
        else:
            self.noise_std = 0.
        self.mask_ratio = args.mask_ratio
        self.signal_length = args.signal_length

    def get_pretrain_data(self):
        train_data = []
        train_labels = []
        train_SNR = []

        # Currently, we directly load all the data, we may try the lazy load method to save the memory.
        for dataset in self.args.dataset:
            dataset = build_dataset(dataset, self.args.dataset_path)(test_size=self.args.test_size, val_size=self.args.val_size)
            IQ_data, label, SNR = dataset.get_pretrain_data()
            self.classes = dataset.classes
            # IQ_data, label, SNR = build_dataset(dataset, self.args.test_size, self.args.dataset_path)().get_pretrain_data
            train_data.extend(IQ_data)
            train_labels.extend(label)
            train_SNR.extend(SNR)

        train_dataset = MaskedDataset(train_data, train_labels, train_SNR, self.noise_std, self.mask_ratio)
        return train_dataset

    def get_loss_func(self):
        return F.mse_loss
    
    def get_data(self):
        dataset = build_dataset(self.args.dataset[0], self.args.dataset_path)
        processed_dataset = dataset(self.args.test_size)
        self.classes = dataset.classes

        return processed_dataset
    
    def get_classes(self):
        return self.classes
