import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from ..tasks import build_task
from ..models import build_model
from . import register_pipe
from .base_pipe import BasePipe
from ..utils.early_stop import EarlyStopping
from ..utils.plot import plot_confusion_matrix, plot_tsne

@register_pipe("spectrumfm_tune")
class SpectrumFMTune(BasePipe):
    def __init__(self, args):
        super(SpectrumFMTune, self).__init__(args)
        self.model_name = args.model
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

        task_name = args.task
        self.task = build_task(args, task_name)
        self.loss_fn = self.task.get_loss_func()
        train_dataset, val_dataset, test_dataset = self.task.get_data()
        self.classes = self.task.get_classes()
        self.gru = nn.GRU(args.hidden_dim, args.hidden_dim, batch_first=True, bidirectional=True).to(args.device)
        self.fc = nn.Linear(args.hidden_dim * 2, len(self.classes)).to(args.device)
        self.dropout = nn.Dropout(0.2).to(args.device)
        self.optimizer = self.candidate_optimizer[args.optimizer]([{"params": self.model.parameters()},
                                                                    {"params": self.gru.parameters()},
                                                                    {"params": self.fc.parameters()}],
                                                                    lr=args.lr, weight_decay=args.weight_decay)

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        self.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

        self.scaler = torch.cuda.amp.GradScaler()
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=3, verbose=True, min_lr=5e-5)
        self.SNR_list = self.task.get_snr()
        self.output_dir = args.output_dir
        if not hasattr(args, "plot"):
            self.plot = False
        else:
            self.plot = args.plot

    def train(self):
        stopper = EarlyStopping(self.args.patience, self._checkpoint, self.args.compile_flag, self.args.use_distribute)
        iters_per_epoch = len(self.train_loader)
        for epoch in range(self.args.num_epochs):
            self.model.train()
            train_loss, train_acc, train_true, train_pred = self._train_step()
            print(f"Epoch:{epoch}, train_loss={train_loss}, train_acc={train_acc['Avg']}")
            print(train_acc)
            if self.plot:
                plot_confusion_matrix(train_true, train_pred, self.dataset_name, "all", self.output_dir, self.classes)

            if epoch % self.args.evaluate_interval == 0:
                loss, acc, true, pred, val_snr = self._test_step("val")
                print(f"Epoch:{epoch}, val_loss={loss}, val_acc={acc['Avg']}")
                early_stop = stopper.loss_step(loss, self.model)
                if self.plot:
                    plot_confusion_matrix(true, pred, self.dataset_name, "all", self.output_dir, self.classes)

            if early_stop:
                print("Early Stop!\tEpoch:" + str(epoch))
                break

        stopper.load_model(self.model)
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
                # modacc = pd.DataFrame.from_dict(mod_dic, orient='index', columns=classes).reset_index(names='SNR')
                # modacc.to_csv(f'logs/{model_tag}/Test_mod_SNR.csv', index=False)
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
        for _, data in enumerate(tqdm(self.train_loader)):
            batch_x, batch_y, batch_SNR = data
            num_sample = batch_x.size(0)
            num_total += num_sample
            batch_SNR = batch_SNR.numpy().tolist()
            batch_y = batch_y.to(self.device)
            batch_x = batch_x.transpose(1, 2).to(self.device)
            if num_sample < self.args.batch_size:
                padding = torch.zeros((self.args.batch_size - num_sample, self.args.signal_length, 2)).to(batch_x.device)
                batch_x = torch.cat((batch_x, padding), dim=0)
            emb = self.model(batch_x)
            output, _ = self.gru(emb)
            output = output[:, -1, :]
            output = self.dropout(output)
            batch_out = self.fc(output)[:num_sample]

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
                batch_SNR = batch_SNR.numpy().tolist()
                batch_y = batch_y.to(self.device)
                batch_x = batch_x.transpose(1, 2).to(self.device)
                if num_sample < self.args.batch_size:
                    padding = torch.zeros((self.args.batch_size - num_sample, self.args.signal_length, 2)).to(batch_x.device)
                    batch_x = torch.cat((batch_x, padding), dim=0)
                emb = self.model(batch_x)
                output, _ = self.gru(emb)
                output = output[:, -1, :]
                output = self.dropout(output)
                batch_out = self.fc(output)[:num_sample]
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
