import os
import os.path as osp

from typing import Union, List
from .utils import set_random_seed, load_yaml, set_random_seed, Config
from .pipeline import build_pipe
import torch

class Experiment(object):
    def __init__(self, model: str,
                dataset: Union[str, List[str]],
                task: str,
                gpu: Union[int, List[int]] = -1,
                load_from_pretrained: bool = False,
                conf_path: str = None,
                hpo_search_space = None,
                hpo_trials = 10,
                **kwargs):

        if conf_path is None:
            config_path = osp.join(osp.join(osp.join(osp.dirname(osp.abspath(__file__)), "models"), model), "config.yaml")
        else:
            config_path = conf_path
        if osp.exists(config_path):
            self.config = load_yaml(config_path)
        else:
            self.config = {}

        self.config = Config(**self.config)
        self.config.model = model
        self.config.dataset = dataset
        self.config.task = task
        self.config.hpo_search_space = hpo_search_space
        self.config.hpo_trials = hpo_trials
        self.set_params(**kwargs)

        if not hasattr(self.config, "dataset_path"):
            self.config.dataset_path = None
        elif self.config.dataset_path == "None":
            self.config.dataset_path = None

        if isinstance(gpu, int):
            self.config.use_distribute = False
            if gpu < 0:
                self.config.device = "cpu"
            else:
                self.config.device = torch.device("cuda", gpu)
        else:
            self.num_gpus = len(gpu)
            self.config.use_distribute = True

        self.config.load_from_pretrained = load_from_pretrained
        self.repository_dir = osp.dirname(osp.abspath(__file__))
        self.config.output_dir = osp.join(self.repository_dir, "output", self.config.model)
        os.makedirs(self.config.output_dir, exist_ok=True)

        if not getattr(self.config, "seed", False):
            self.config.seed = 0

    def set_params(self, **kwargs):
        for key, value in kwargs.items():
            self.config.__setattr__(key, value)

    def run(self):
        if hasattr(self, "num_gpus"):
            import socketserver
            import torch.multiprocessing as mp
            with socketserver.TCPServer(("localhost", 0), None) as s:
                port_id = s.server_address[1]
            mp.spawn(self.distribute_run, args=(port_id, ), nprocs=self.num_gpus)
            return
        set_random_seed(self.config.seed)
        supported_pipeline_path = osp.join(osp.dirname(osp.abspath(__file__)), "supported.yaml")
        supported_pipeline = load_yaml(supported_pipeline_path)
        pipe = supported_pipeline.get(self.config.model)
        if not isinstance(pipe, str):
            pipe = pipe.get(self.config.task)

        if self.config.hpo_search_space is not None:
            from .auto import hpo_experiment
            hpo_experiment(self.config, pipe)
        else:
            pipeline = build_pipe(self.config, pipe)
            if "pretrain" in pipeline.__class__.__name__.lower():
                pipeline.train()
            else:
                pipeline.train()

    def distribute_run(self, replica_id, port):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(port)
        torch.distributed.init_process_group(
            "nccl", rank=replica_id, world_size=self.num_gpus)
        self.config.device = torch.device("cuda", replica_id)
        set_random_seed(self.config.seed)
        supported_pipeline_path = osp.join(osp.dirname(osp.abspath(__file__)), "supported.yaml")
        supported_pipeline = load_yaml(supported_pipeline_path)
        pipe = supported_pipeline.get(self.config.model)
        if not isinstance(pipe, str):
            pipe = pipe.get(self.config.task)
        pipeline = build_pipe(self.config, pipe)
        if "pretrain" in pipeline.__class__.__name__.lower():
            pipeline.train()
        else:
            pipeline.train()
        

    def __repr__(self):
        basic_info = "------------------------------------------------------------------------------\n" \
                     " Basic setup of this experiment: \n" \
                     "     model: {}    \n" \
                     "     dataset: {}   \n" \
                     " This experiment has following parameters. You can use set_params to edit them.\n" \
                     " Use print(experiment) to print this information again.\n" \
                     "------------------------------------------------------------------------------\n". \
            format(self.config.model, self.config.dataset)
        params_info = ""
        for attr in dir(self.config):
            if "__" not in attr and attr not in self.immutable_params:
                params_info += "{}: {}\n".format(attr, getattr(self.config, attr))
        return basic_info + params_info
