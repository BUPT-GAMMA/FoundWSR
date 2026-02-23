import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from ..tasks import build_task
from ..models import build_model
from . import register_pipe
from .base_pipe import BasePipe
from ..utils import Time_Freq_Diffusion
from ..utils.early_stop import EarlyStopping

@register_pipe("tsddae_pretrain")
class TSDDAEPretrain(BasePipe):
    def __init__(self, args):
        super(TSDDAEPretrain, self).__init__(args)
        self.model = build_model(args.model).build_model_from_args(args).to(args.device)
        self.norm_epsilon = nn.LayerNorm(args.hidden_dim).to(args.device)
        self.norm_eta = nn.LayerNorm(args.hidden_dim).to(args.device)
        self.act = nn.SiLU().to(args.device)
        self.epsilon = nn.Linear(args.hidden_dim, 2).to(args.device)
        self.eta = nn.Linear(args.hidden_dim, 2).to(args.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')
        if args.load_from_pretrained:
            self.load_from_pretrained()
        if hasattr(args, "compile_flag"):
            if args.compile_flag:
                self.compile()
        if args.use_distribute:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.device])
        print("-----------------------load model done-----------------------")
        self.max_step = args.max_step
        self.optimizer = self.candidate_optimizer[args.optimizer]([{"params": self.model.parameters()},
                                                                   {"params": self.norm_epsilon.parameters()},
                                                                   {"params": self.norm_eta.parameters()},
                                                                   {"params": self.epsilon.parameters()},
                                                                   {"params": self.eta.parameters()}],
                                                                  lr=args.lr, weight_decay=args.weight_decay)

        task_name = "signal_prediction"
        self.pretrain_task = build_task(args, task_name)
        self.pretrain_loss_fn = self.pretrain_task.get_loss_func()
        train_dataset, length_list = self.pretrain_task.get_pretrain_data()
        self.dataloader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       pin_memory=False if args.device=="cpu" else True,
                                       sampler=DistributedSampler(train_dataset) if args.use_distribute else None,
                                       drop_last=True)

        self.diffusion = Time_Freq_Diffusion(self.max_step, self.args.min_noise, self.args.max_noise, args.ratio, args.device)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=250, verbose=True, min_lr=1e-12)

    @torch._dynamo.disable
    def get_fft_input(self, data):
        I = data[:, 0, :]
        Q = data[:, 1, :]
        s = torch.complex(I, Q)
        freq = torch.fft.fft(s)
        re = torch.real(freq)
        im = torch.imag(freq)
        fft_data = torch.concat([re.unsqueeze(1), im.unsqueeze(1)], dim=1)
        return fft_data

    def train(self):
        stopper = EarlyStopping(self.args.patience, self._checkpoint, 
                                self.args.compile_flag, self.args.use_distribute)
        self.model.train()
        early_stop = torch.tensor(False, dtype=torch.bool, device=self.args.device)
        loss_list = []
        for epoch in range(15):
            for i, (data, _) in enumerate(tqdm(self.dataloader)):
                data = data.to(self.device, non_blocking=True)
                max_abs = data.abs().amax(dim=-1, keepdim=True)
                data = data / max_abs
                t = torch.randint(0, self.max_step, (data.shape[0], ), dtype=torch.int64).to(self.device)
                x_noised, epsilon, eta, ratio = self.diffusion.q_sample(data, t)
                fft_data = self.get_fft_input(x_noised)
                out1, out2 = self.model(x_noised, fft_data, t / self.max_step)
                pred_epsilon = self.epsilon(self.act(self.norm_epsilon(out1))).transpose(1, 2)
                pred_eta = self.eta(self.act(self.norm_eta(out2))).transpose(1, 2)
                loss = self.pretrain_loss_fn(epsilon + ratio * eta, pred_epsilon + ratio * pred_eta)
                print(loss)
                loss_list.append(loss.detach().cpu().numpy())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step(loss.item())
                
                if i % self.args.evaluate_interval == 0:
                    print(f"---loss:{loss.item()}---lr:{self.optimizer.param_groups[0]['lr']}---")
                early_stop = torch.tensor(stopper.loss_step(loss, self.model), dtype=torch.bool, device=self.args.device)
                if early_stop:
                    break

            if early_stop:
                break

        stopper.load_model(self.model)