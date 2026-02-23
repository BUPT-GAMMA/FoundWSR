import torch.nn.functional as F
from . import BaseTask, register_task
from ..dataset import build_dataset

@register_task("classification")
class Classification(BaseTask):
    def __init__(self, args):
        super(Classification, self).__init__()
        self.args = args
        self.signal_length = args.signal_length

    def get_data(self):
        dataset = build_dataset(self.args.dataset[0], self.args.dataset_path)
        self.classes = dataset.classes
        params = {
            attr: getattr(self.args, attr)
            for attr in ['test_size', 'val_size']
            if hasattr(self.args, attr)
        }
        train_dataset = dataset("train", **params)
        val_dataset = dataset("valid", **params)
        test_dataset = dataset("test", **params)
        self.minSNR = train_dataset.minSNR
        self.maxSNR = train_dataset.maxSNR
        self.SNR_list = train_dataset.SNR_list

        return train_dataset, val_dataset, test_dataset

    def get_loss_func(self):
        return F.cross_entropy
    
    def get_classes(self):
        return self.classes

    def get_snr(self):
        return list(self.SNR_list)
