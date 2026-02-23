import math
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from ..tasks import build_task
from ..models import build_model
from . import register_pipe
from .base_pipe import BasePipe
from ..utils.early_stop import EarlyStopping

@register_pipe("moco_pretrain")
class MoCoPretrain(BasePipe):
    def __init__(self, args):
        super(MoCoPretrain, self).__init__(args)
        self.model = build_model(args.model).build_model_from_args(args).to(args.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        if hasattr(args, "compile_flag"):
            if args.compile_flag:
                self.compile()
        if args.load_from_pretrained:
            self.load_from_pretrained()
        print("-----------------------load model done-----------------------")
        self.optimizer = self.candidate_optimizer[args.optimizer](self.model.parameters(),
                                                                  lr=args.lr, weight_decay=args.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler()
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3, verbose=True, min_lr=5e-5)
        task_name = "signal_prediction"
        self.task = build_task(args, task_name)
        train_dataset = self.task.get_pretrain_data()
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       pin_memory=False if args.device=="cpu" else True,
                                       sampler=DistributedSampler(train_dataset) if args.use_distribute else None,
                                       drop_last=True)
    def train(self):
        stopper = EarlyStopping(self.args.patience, self._checkpoint)
        self.model.train()
        iters_per_epoch = len(self.train_loader)
        for epoch in range(self.args.num_epochs):
            for i, (IQ_original, IQ_agumented, stp_original, stp_agumented) in enumerate(self.train_loader):
                if self.args.moco_m_cos:
                    moco_m = self.adjust_moco_momentum(epoch + i / iters_per_epoch, self.args)
                else:
                    moco_m = 1.

                IQ_original = IQ_original.to(self.device, non_blocking=True)
                IQ_agumented = IQ_agumented.to(self.device, non_blocking=True)
                stp_original = stp_original.to(self.device, non_blocking=True)
                stp_agumented = stp_agumented.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(True):
                    loss = self.model(IQ_original, IQ_agumented, stp_original, stp_agumented, moco_m)
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step(loss.item())
                if i % self.args.evaluate_interval == 0:
                    print(f"---loss:{loss.item()}---lr:{self.optimizer.param_groups[0]['lr']}---")
                early_stop = stopper.loss_step(loss, self.model)
                if early_stop:
                    print("Early Stop!\tEpoch:" + str(epoch))
                    break
            if early_stop:
                break

    @staticmethod
    def adjust_moco_momentum(epoch, args):
        """Adjust moco momentum based on current epoch"""
        m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.num_epochs)) * (1. - args.moco_m)
        return m
