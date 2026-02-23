import copy
import os.path as osp
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

@register_pipe("tsddae_tune")
class TSDDAETune(BasePipe):
    def __init__(self, args):
        super(TSDDAETune, self).__init__(args)
        self.model = build_model(args.model).build_model_from_args(args).to(args.device)
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

        if args.task == "amc" or args.task == "wtc":
            task_name = "classification"
        self.task = build_task(args, task_name)
        self.loss_fn = self.task.get_loss_func()
        train_dataset, val_dataset, test_dataset = self.task.get_data()
        self.classes = self.task.get_classes()
        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        self.diffusion = Time_Freq_Diffusion(self.max_step, self.args.min_noise, self.args.max_noise, args.ratio, args.device)

        self.classifier = nn.ModuleList([nn.AdaptiveAvgPool1d(1),
                                         nn.AdaptiveAvgPool1d(1),
                                         nn.Sequential(
                                                        nn.Linear(2 * args.hidden_dim, args.hidden_dim),
                                                        nn.Dropout(0.2),
                                                        nn.PReLU(),
                                                        nn.Linear(args.hidden_dim, len(self.classes)))]).to(args.device)

        self.optimizer = self.candidate_optimizer[args.optimizer]([{"params": self.model.parameters()},
                                                                             {"params": self.classifier.parameters()}],
                                                                            lr=1e-3, weight_decay=args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)

        self.SNR_list = self.task.get_snr()
        self.output_dir = args.output_dir

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
            batch_x, batch_y, batch_SNR = data
            num_sample = batch_x.size(0)
            num_total += num_sample
            batch_SNR = batch_SNR.numpy().tolist()
            batch_y = batch_y.to(self.device)
            batch_x = batch_x.to(self.device)
            max_abs = batch_x.abs().amax(dim=-1, keepdim=True)
            batch_x = batch_x / max_abs
            t = torch.tensor([self.args.timestep], dtype=torch.int64).to(self.device)

            x_noised, epsilon, eta, ratio = self.diffusion.q_sample(batch_x, t)
            fft_data = self.get_fft_input(x_noised)
            out1, out2 = self.model(x_noised, fft_data, t / self.max_step)
            batch_out1 = self.classifier[0](out1.transpose(1, 2)).squeeze(-1)
            batch_out2 = self.classifier[1](out2.transpose(1, 2)).squeeze(-1)
            batch_out = torch.concat([batch_out1, batch_out2], dim=-1)
            batch_out = self.classifier[2](batch_out)
            batch_loss = self.loss_fn(batch_out, batch_y)
            print(batch_loss)

            train_pred = batch_out.cpu().detach().numpy()
            train_pred = train_pred.argmax(1).tolist()
            train_true = batch_y.cpu().detach().numpy().tolist()

            y_true.extend(train_true)
            y_pred.extend(train_pred)

            for slice in range(num_sample):
                if isinstance(batch_SNR[slice], list):
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
                batch_x, batch_y, batch_SNR = data
                num_sample = batch_x.size(0)
                num_total += num_sample
                batch_x = batch_x.to(self.device)       
                batch_SNR = batch_SNR.numpy().tolist()
                batch_y = batch_y.to(self.device)
                batch_x = batch_x.to(self.device)
                max_abs = batch_x.abs().amax(dim=-1, keepdim=True)
                batch_x = batch_x / max_abs
                t = torch.tensor([self.args.timestep], dtype=torch.int64).to(self.device)
                x_noised, epsilon, eta, ratio = self.diffusion.q_sample(batch_x, t)
                fft_data = self.get_fft_input(x_noised)
                out1, out2 = self.model(x_noised, fft_data, t / self.max_step)
                batch_out1 = self.classifier[0](out1.transpose(1, 2)).squeeze(-1)
                batch_out2 = self.classifier[1](out2.transpose(1, 2)).squeeze(-1)
                batch_out = torch.concat([batch_out1, batch_out2], dim=-1)
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