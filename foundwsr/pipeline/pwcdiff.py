import os.path as osp
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from ..tasks import build_task
from ..models import build_model
from . import register_pipe
from .base_pipe import BasePipe
from ..utils import Diffusion
# from ..utils.sampler import LengthBucketSampler
from ..utils.early_stop import EarlyStopping
from ..utils import plot_confusion_matrix, plot_tsne
from ..utils import channel_corrupt
from tqdm import tqdm
import matplotlib.pyplot as plt

DATASET_CONFIG = {
    "RML2016.10a": ["rician", "cfo", "awgn", "sro"],
    "HisarMod2019": ["rician", "cfo", "multipath", "sro", "awgn"],
    "Techrec": ["awgn"],
    "GNSS": ["rician", "cfo", "awgn", "sro"],
}

@register_pipe("PWCDiff")
class PWCDiff(BasePipe):
    def __init__(self, args):
        super(PWCDiff, self).__init__(args)
        task_name = "signal_prediction"
        self.pretrain_task = build_task(args, task_name)
        self.pretrain_loss_fn = self.pretrain_task.get_loss_func()
        train_dataset = self.pretrain_task.get_pretrain_data()
        self.dataloader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       pin_memory=False if args.device=="cpu" else True,
                                       sampler=DistributedSampler(train_dataset) if args.use_distribute else None,
                                       drop_last=True)
        task_name = "classification"
        self.task = build_task(args, task_name)
        self.loss_fn = self.task.get_loss_func()
        train_dataset, val_dataset, test_dataset = self.task.get_data()
        self.classes = self.task.get_classes()
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        channel_type = DATASET_CONFIG[args.dataset[0]]
        setattr(args, "channel_type", channel_type)

        self.model = build_model(args.model).build_model_from_args(args).to(args.device)
        self.norm_epsilon = nn.LayerNorm(args.hidden_dim).to(args.device)
        self.act = nn.SiLU().to(args.device)
        self.epsilon = nn.Linear(args.hidden_dim, 2).to(args.device)

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
        self.pretrain_optimizer = self.candidate_optimizer[args.optimizer]([{"params": self.model.parameters()},
                                                                   {"params": self.norm_epsilon.parameters()},
                                                                   {"params": self.epsilon.parameters()}],
                                                                  lr=args.lr, weight_decay=args.weight_decay)

        self.diffusion = Diffusion(self.max_step, self.args.min_noise, self.args.max_noise, args.device)
        self.pretrain_scheduler = ReduceLROnPlateau(self.pretrain_optimizer, 'min', factor=0.5, patience=200, verbose=True, min_lr=1e-12)
        self.classifier = nn.ModuleList([nn.AdaptiveAvgPool1d(1),
                                         nn.AdaptiveAvgPool1d(1),
                                         nn.Sequential(
                                                        nn.Linear(args.hidden_dim, args.hidden_dim),
                                                        nn.Dropout(0.2),
                                                        nn.PReLU(),
                                                        nn.Linear(args.hidden_dim, len(self.classes)))]).to(args.device)

        self.optimizer = self.candidate_optimizer[args.optimizer]([{"params": self.model.parameters()},
                                                                             {"params": self.classifier.parameters()}],
                                                                            lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)
        self.SNR_list = self.task.get_snr()
        self.checkpoint = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), f"output/signal/signal_{self.args.dataset[0]}_pretrain.pt")

        if not hasattr(args, "plot"):
            self.plot = False
        else:
            self.plot = args.plot

    @torch._dynamo.disable
    def get_fft_input(self, data):
        I = data[:, 0, :]
        Q = data[:, 1, :]
        s = torch.complex(I, Q)
        fft_data = torch.fft.fft(s, dim=-1)
        re = torch.real(fft_data)
        im = torch.imag(fft_data)
        fft_data = torch.concat([re.unsqueeze(1), im.unsqueeze(1)], dim=1)
        return fft_data

    @torch._dynamo.disable
    def get_ifft_input(self, data):
        amp = data[:, 0, :]
        phase = data[:, 1, :]
        s = torch.complex(amp, phase)
        iq = torch.fft.ifft(s, dim=-1)
        re = torch.real(iq)
        im = torch.imag(iq)
        iq_data = torch.concat([re.unsqueeze(1), im.unsqueeze(1)], dim=1)
        return iq_data

    def train(self):
        stopper = EarlyStopping(self.args.patience, self.checkpoint, 
                                self.args.compile_flag, self.args.use_distribute)
        self.model.train()
        early_stop = torch.tensor(False, dtype=torch.bool, device=self.args.device)
        loss_list = []

        for epoch in range(15):
            for i, (data, _, _, _) in enumerate(tqdm(self.dataloader)):
                data = data.to(self.device, non_blocking=True)
                max_abs = data.abs().amax(dim=-1, keepdim=True)
                data = data / max_abs
                t = torch.randint(0, self.max_step, (data.shape[0], ), dtype=torch.int64).to(self.device)
                channel_data = self.model.get_input(data)
                x_noised, epsilon = self.diffusion.q_sample(channel_data, t)
                iq_data = self.get_ifft_input(x_noised)
                out = self.model(iq_data, x_noised, t / self.max_step)
                pred_data = self.epsilon(self.act(self.norm_epsilon(out))).transpose(1, 2)
                loss = self.pretrain_loss_fn(data,  pred_data)
                loss_list.append(loss.detach().cpu().numpy())
                self.pretrain_optimizer.zero_grad()
                loss.backward()
                self.pretrain_optimizer.step()
                self.pretrain_scheduler.step(loss.item())
                
                if i % self.args.evaluate_interval == 0:
                    print(f"---loss:{loss.item()}---lr:{self.pretrain_optimizer.param_groups[0]['lr']}---")
                early_stop = torch.tensor(stopper.loss_step(loss, self.model), dtype=torch.bool, device=self.args.device)
                if early_stop:
                    break

            if early_stop:
                break
        stopper.load_model(self.model)

        stopper = EarlyStopping(self.args.tune_patience, self._checkpoint,
                                self.args.compile_flag, self.args.use_distribute)
        iters_per_epoch = len(self.train_loader)
        best_loss = None
        for epoch in range(self.args.tune_epochs):
            self.model.train()
            self.classifier.train()
            train_loss, train_acc, train_true, train_pred = self._train_step()
            print(f"Epoch:{epoch}, train_loss={train_loss}, train_acc={train_acc['Avg']}")
            print(train_acc)
            if self.plot:
                plot_confusion_matrix(train_true, train_pred, self.dataset_name, "all", self.output_dir, self.classes)
            if epoch % self.args.evaluate_interval == 0:
                loss, acc, true, pred, val_snr = self._test_step("val")
                print(f"Epoch:{epoch}, val_loss={loss}, val_acc={acc['Avg']}")
                early_stop = stopper.loss_step(loss, self.model)
                if isinstance(loss, torch.Tensor):
                    loss = loss.item()
                if best_loss is None:
                    best_loss = loss
                    best_classifier = copy.deepcopy(self.classifier)
                else:
                    if loss < best_loss:
                        best_classifier = copy.deepcopy(self.classifier)
                    best_loss = np.min((loss, best_loss))
                
                if self.plot:
                    plot_confusion_matrix(true, pred, self.dataset_name, "all", self.output_dir, self.classes)

            if early_stop:
                print("Early Stop!\tEpoch:" + str(epoch))
                break

        stopper.load_model(self.model)
        self.classifier = best_classifier
        classifier_path = osp.join(osp.dirname(self._checkpoint), "classifier_" + self.args.dataset[0] + ".pt")
        torch.save(self.classifier.state_dict(), classifier_path)
        # self.classifier.load_state_dict(torch.load(classifier_path), strict=False)
        test_loss, test_acc, test_true, test_pred, test_SNR = self._test_step("test")
        performance = test_acc["Avg"]
        print(f"test acc={performance}")
        print(f"performance under various SNR: {test_acc}")

        if self.plot:
            mod_dic = {}
            for snr in self.SNR_list:
                SNR_cm = [i for i in zip(test_SNR, pred, true) if i[0] == snr]
                true_cm = []
                pred_cm = []
                true_cls = np.zeros(len(self.classes))
                all = np.zeros(len(self.classes))
                for i in SNR_cm:
                    pred_cm.append(i[1])
                    true_cm.append(i[2])
                    if i[1] == i[2]:
                        true_cls[i[1]] = true_cls[i[1]] + 1
                        all[i[1]] = all[i[1]] + 1
                    else:
                        all[i[2]] = all[i[2]] + 1
                cls_acc = {cls: x / y for cls, x, y in zip(self.classes, true_cls, all)}
                mod_dic[snr] = list(cls_acc.values())
                plot_confusion_matrix(test_acc, test_true, self.dataset_name, snr, self.output_dir, self.classes)
                SNR_tsne_ = [i for i in
                        zip(test_SNR, torch.stack(test_pred).cpu().data.numpy(), torch.stack(test_true).cpu().data.numpy()) if
                        i[0] == snr]
                _, pred_0, true_0 = zip(*SNR_tsne_)
                plot_tsne(np.array(list(pred_0)), np.array(list(true_0)), self.dataset_name, snr, self.output_dir, self.classes)
        return test_acc

    def _train_step(self):
        SNR = dict([(key, 0) for key in self.SNR_list])
        SNR_true = dict([(key, 0) for key in self.SNR_list])
        y_true = []
        y_pred = []
        num_total = 0
        loss = 0.0
        ssim_list = []
        for i, data in enumerate(tqdm(self.train_loader)):
            batch_x, _, batch_y, batch_SNR = data
            num_sample = batch_x.size(0)
            num_total += num_sample
            batch_SNR = batch_SNR.numpy().tolist()
            batch_y = batch_y.to(self.device)
            batch_x = batch_x.to(self.device)
            max_abs = batch_x.abs().amax(dim=-1, keepdim=True)
            batch_x = batch_x / max_abs
            t = torch.tensor([self.args.timestep], dtype=torch.int64).to(self.device)
            fft_data = self.get_fft_input(batch_x)
            out1 = self.model(batch_x, fft_data, t / self.max_step)
            batch_out = self.classifier[0](out1.transpose(1, 2)).squeeze(-1)
            batch_out = self.classifier[2](batch_out)
            batch_loss = self.loss_fn(batch_out, batch_y)
            print(batch_loss)

            train_pred = batch_out.cpu().detach().numpy()
            train_pred = train_pred.argmax(1).tolist()
            train_true = batch_y.cpu().detach().numpy().tolist()

            y_true.extend(train_true)
            y_pred.extend(train_pred)

            for slice in range(num_sample):
                if (type(batch_SNR[slice])).__name__ == 'list':
                    batch_SNR[slice] = batch_SNR[slice][0]
                if train_pred[slice] == train_true[slice]:
                    SNR[batch_SNR[slice]] = SNR.get(batch_SNR[slice]) + 1
                    SNR_true[batch_SNR[slice]] = SNR_true.get(batch_SNR[slice]) + 1
                else:
                    SNR[batch_SNR[slice]] = SNR.get(batch_SNR[slice]) + 1

            loss += (batch_loss.item() * num_sample)
            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()
            if i % self.args.evaluate_interval == 0:
                print(f"---loss:{batch_loss.item()}---lr:{self.optimizer.param_groups[0]['lr']}---")

        loss /= num_total
        avg_true = 0
        avg_all = 0
        for key in self.SNR_list:
            avg_all += SNR[key]
            avg_true += SNR_true[key]
            SNR[key] = SNR_true[key] / float(SNR[key])
        SNR['Avg'] = avg_true / float(avg_all)

        
        return loss, SNR, y_true, y_pred

    def _test_step(self, mode):
        self.model.eval()
        self.classifier.eval()
        SNR = dict([(key, 0) for key in self.SNR_list])
        SNR_true = dict([(key, 0) for key in self.SNR_list])
        y_true = []
        y_pred = []
        eval_SNR = []

        num_total = 0
        loss = 0.0
        if "val" in mode:
            loader = self.val_loader
        elif "test" in mode:
            loader = self.test_loader

        with torch.no_grad():
            for _, data in enumerate(loader):
                batch_x, _, batch_y, batch_SNR = data
                num_sample = batch_x.size(0)
                num_total += num_sample
                batch_x = batch_x.to(self.device)       
                batch_SNR = batch_SNR.numpy().tolist()
                batch_y = batch_y.to(self.device)
                batch_x = batch_x.to(self.device)
                max_abs = batch_x.abs().amax(dim=-1, keepdim=True)
                batch_x = batch_x / max_abs
                t = torch.tensor([self.args.timestep], dtype=torch.int64).to(self.device)
                fft_data = self.get_fft_input(batch_x)
                out1 = self.model(batch_x, fft_data, t / self.max_step)
                batch_out = self.classifier[0](out1.transpose(1, 2)).squeeze(-1)
                batch_out = self.classifier[2](batch_out)
                batch_loss = self.loss_fn(batch_out, batch_y)

                train_pred = batch_out.cpu().detach().numpy()
                train_pred = train_pred.argmax(1).tolist()
                train_true = batch_y.cpu().detach().numpy().tolist()
                y_true.extend(train_true)
                y_pred.extend(train_pred)
                eval_SNR.extend(batch_SNR)

                for slice in range(num_sample):
                    if isinstance(batch_SNR[slice], list):
                        batch_SNR[slice] = batch_SNR[slice][0]
                    if train_pred[slice] == train_true[slice]:
                        SNR[batch_SNR[slice]] = SNR.get(batch_SNR[slice]) + 1
                        SNR_true[batch_SNR[slice]] = SNR_true.get(batch_SNR[slice]) + 1
                    else:
                        SNR[batch_SNR[slice]] = SNR.get(batch_SNR[slice]) + 1

                loss += (batch_loss.item() * num_sample)

        loss /= num_total
        self.scheduler.step(loss)
        avg_true = 0
        avg_all = 0
        for key in self.SNR_list:
            avg_all += SNR[key]
            avg_true += SNR_true[key]
            SNR[key] = SNR_true[key] / float(SNR[key])
        SNR['Avg'] = avg_true / float(avg_all)
        
        return loss, SNR, y_true, y_pred, eval_SNR

