import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from ..tasks import build_task
from ..models import build_model
from . import register_pipe
from .base_pipe import BasePipe
from ..utils.early_stop import EarlyStopping

def create_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
    return lr_lambda

@register_pipe("spectrumfm_pretrain")
class SpectrumFMPretrain(BasePipe):
    def __init__(self, args):
        super(SpectrumFMPretrain, self).__init__(args)
        self.model = build_model(args.model).build_model_from_args(args).to(args.device)
        self.construction = nn.Linear(args.hidden_dim, args.input_dim).to(args.device)
        self.gru = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True).to(args.device)
        self.fc = nn.Linear(args.hidden_dim, 2).to(args.device)
        self.dropout = nn.Dropout(0.2).to(args.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")
        if args.load_from_pretrained:
            self.load_from_pretrained()
        if hasattr(args, "compile_flag"):
            if args.compile_flag:
                self.compile()
        if args.use_distribute:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.device])
            self.construction = nn.parallel.DistributedDataParallel(self.construction, device_ids=[args.device])
            self.gru = nn.parallel.DistributedDataParallel(self.gru, device_ids=[args.device])
            self.fc = nn.parallel.DistributedDataParallel(self.fc, device_ids=[args.device])
        print("-----------------------load model done-----------------------")
        self.optimizer = self.candidate_optimizer[args.optimizer]([{"params": self.model.parameters()}, 
                                                                   {"params": self.construction.parameters()},
                                                                   {"params": self.gru.parameters()},
                                                                   {"params": self.fc.parameters()}],
                                                                  lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, create_lr_lambda(args.warmup_steps, args.total_steps))
        reconstruct_task_name = "masked_reconstruction"
        self.reconstruct_task = build_task(args, reconstruct_task_name)
        self.reconstruct_loss_fn = self.reconstruct_task.get_loss_func()
        train_dataset = self.reconstruct_task.get_pretrain_data()
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       pin_memory=False if args.device=="cpu" else True,
                                       sampler=DistributedSampler(train_dataset) if args.use_distribute else None,
                                       drop_last=True)
        predict_task_name = "signal_prediction"
        self.predict_task = build_task(args, predict_task_name)
        self.predict_loss_fn = self.predict_task.get_loss_func()

    def train(self):
        stopper = EarlyStopping(self.args.patience, self._checkpoint, self.args.compile_flag, self.args.use_distribute)
        self.model.train()
        for epoch in range(self.args.num_epochs):
            for i, (data, _, _, masked_data, mask, pre_label) in enumerate(self.train_loader):
                data = data.to(self.device)
                masked_data = masked_data.to(self.device)
                mask = mask.to(self.device)
                pre_label = pre_label.to(self.device)

                emb = self.model(masked_data, mask)
                constructed = self.construction(emb)
                output, _ = self.gru(emb[:, :-1, :])
                output = output[:, -1, :]
                output = self.dropout(output)
                predicted = self.fc(output)

                mask_expanded = mask.unsqueeze(-1).expand_as(data)
                mask_bool = mask_expanded == 0
                y_true_masked = data[mask_bool]
                y_pred_masked = constructed[mask_bool]
                if y_true_masked.numel() == 0:
                    reconstructed_loss = torch.tensor(0., device=self.device)
                else:
                    reconstructed_loss = self.reconstruct_loss_fn(y_true_masked, y_pred_masked, reduction="mean")

                predicted_loss = self.predict_loss_fn(predicted, pre_label)
                loss = reconstructed_loss + predicted_loss
                print(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss.item())
                if i % self.args.evaluate_interval == 0:
                    print(f"---loss:{loss.item()}---lr:{self.optimizer.param_groups[0]['lr']}---")
                early_stop = stopper.loss_step(loss, self.model)
                if early_stop:
                    print("Early Stop!\tEpoch:" + str(epoch))
                    break
            print(f"***Epoch:{epoch}***loss:{loss.item()}***lr:{self.optimizer.param_groups[0]['lr']}***")
            if early_stop:
                break
