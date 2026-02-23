import torch.nn.functional as F
from . import BaseTask, register_task
from ..dataset import build_dataset
from ..dataset.pretrain import PretrainDataset

@register_task("signal_prediction")
class SignalPrediction(BaseTask):
    def __init__(self, args):
        super(SignalPrediction, self).__init__()
        self.args = args
        self.signal_length = args.signal_length

    def get_pretrain_data(self):
        train_data = []
        train_labels = []
        train_SNR = []
        length = []
        params = {
            attr: getattr(self.args, attr)
            for attr in ['test_size', 'val_size']
            if hasattr(self.args, attr)
        }
        # Currently, we directly load all the data, we may try the lazy load method to save the memory.
        for dataset_name in self.args.dataset:
            dataset = build_dataset(dataset_name, self.args.dataset_path)
            signal_length = dataset.signal_length
            IQ_data, label, SNR = dataset(**params).get_pretrain_data()
            train_data.extend(IQ_data)
            train_labels.extend(label)
            train_SNR.extend(SNR)
            length.extend([signal_length] * len(IQ_data))
        train_dataset = PretrainDataset(train_data, train_labels, train_SNR)

        return train_dataset, length

    def get_data(self):
        dataset = build_dataset(self.args.dataset[0], self.args.dataset_path)
        processed_dataset = dataset(self.args.test_size)
        self.classes = dataset.classes

        return processed_dataset

    def get_loss_func(self):
        return F.mse_loss

    def get_classes(self):
        return self.classes
