import os.path as osp
import torch
from abc import ABC, abstractmethod

class BasePipe(ABC):
    candidate_optimizer = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "Adadelta": torch.optim.Adadelta,
        "AdamW": torch.optim.AdamW
    }

    def __init__(self, args):
        """

        Parameters
        ----------
        args

        Attributes
        -------------
        evaluate_interval: int
            the interval of evaluation in validation
        """
        super(BasePipe, self).__init__()
        self.evaluator = None
        self.evaluate_interval = getattr(args, "evaluate_interval", 1)

        if "pretrain" in args.task.lower():
            default_path = osp.join(args.output_dir,
                                        f"{args.model}_{args.task}.pt")
        else:
            default_path = osp.join(args.output_dir,
                                        f"{args.model}_{args.dataset[0]}_{args.task}.pt")
        
        
        if hasattr(args, "model_path"):
            self.model_path = args.model_path
        else:
            if hasattr(args, "load_from_pretrained"):
                self.model_path = default_path
            else:
                self.model_path = None

        if hasattr(args, "output_path"):
            self._checkpoint = args.output_path
        else:
            if hasattr(args, "load_from_pretrained"):
                self._checkpoint = default_path
            else:
                self._checkpoint = None

        self.args = args
        self.device = args.device
        self.num_epochs = args.num_epochs
        self.optimizer = None

    @abstractmethod
    def train(self):
        pass

    def load_from_pretrained(self):
        if hasattr(self.args, "load_from_pretrained") and self.args.load_from_pretrained:
            try:
                ck_pt = torch.load(self.model_path)
                self.model.load_state_dict(ck_pt, strict=False)
            except FileNotFoundError:
                print("[Load Model] Do not load the model from pretrained, "
                                      "{} doesn\"t exists".format(self._checkpoint))
        # return self.model

    def save_checkpoint(self):
        if self._checkpoint and hasattr(self.model, "_parameters()"):
            torch.save(self.model.state_dict(), self._checkpoint)

    def compile(self):
        if hasattr(self.args, "compile"):
            self.model = torch.compile(
                self.model, mode=self.args.compile["mode"] if "mode" in self.args.compile else "default",
                fullgraph=self.args.compile["fullgraph"] if "fullgraph" in self.args.compile else False,
                dynamic=self.args.compile["dynamic"] if "dynamic" in self.args.compile else None,
                backend=self.args.compile["backend"] if "backend" in self.args.compile else "inductor"
                )
        else:
            self.model = torch.compile(self.model)
