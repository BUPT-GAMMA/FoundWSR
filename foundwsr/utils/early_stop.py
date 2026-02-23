import numpy as np
import torch
import copy

class EarlyStopping(object):
    def __init__(self, patience=10, save_path=None, compiled=False, distributed=False):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.best_loss = None
        self.early_stop = False
        if save_path is None:
            self.best_model = None
        self.save_path = save_path
        self.compiled = compiled
        self.distributed = distributed

    def step(self, loss, score, model):
        if isinstance(score, tuple):
            score = score[0]
        if self.best_loss is None:
            self.best_score = score
            self.best_loss = loss
            self.save_model(model)
        elif (loss > self.best_loss) and (score < self.best_score):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (score >= self.best_score) and (loss <= self.best_loss):
                self.save_model(model)

            self.best_loss = np.min((loss, self.best_loss))
            self.best_score = np.max((score, self.best_score))
            self.counter = 0
        return self.early_stop

    def step_score(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if score >= self.best_score:
                self.save_model(model)

            self.best_score = np.max((score, self.best_score))
            self.counter = 0
        return self.early_stop

    def loss_step(self, loss, model):
        """

        Parameters
        ----------
        loss Float or torch.Tensor

        model torch.nn.Module

        Returns
        -------

        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        if self.best_loss is None:
            self.best_loss = loss
            self.save_model(model)
        elif loss >= self.best_loss:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if loss < self.best_loss:
                self.save_model(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop

    def save_model(self, model):
        if self.save_path is None:
            self.best_model = copy.deepcopy(model)
        else:
            model.eval()
            if self.distributed:
                if self.compiled:
                    torch.save(model.module._orig_mod.state_dict(), self.save_path)
                else:
                    torch.save(model.module.state_dict(), self.save_path)
            else:
                if self.compiled:
                    torch.save(model._orig_mod.state_dict(), self.save_path)
                else:
                    torch.save(model.state_dict(), self.save_path)
            model.train()

    def load_model(self, model):
        if self.save_path is None:
            return self.best_model
        else:
            model.load_state_dict(torch.load(self.save_path), strict=False)