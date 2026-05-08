import importlib
from .base_pipe import BasePipe
from abc import ABC     

PIPE_REGISTRY = {}
SUPPORTED_PIPES = {
    "moco_pretrain": "foundwsr.pipeline.moco_pretrain",
    "classification": "foundwsr.pipeline.classification",
    "spectrumfm_pretrain": "foundwsr.pipeline.spectrumfm_pretrain",
    "spectrumfm_tune": "foundwsr.pipeline.spectrumfm_tune",
    "iqformer_trainer": "foundwsr.pipeline.iqformer_trainer",
    "tsddae_pretrain": "foundwsr.pipeline.tsddae_pretrain",
    "tsddae_tune": "foundwsr.pipeline.tsddae_tune",
    "few_shot": "foundwsr.pipeline.few_shot",
    "pwcdiff": "foundwsr.pipeline.pwcdiff",
    "prob": "foundwsr.pipeline.prob",
}

def register_pipe(name):
    """
    New pipe can be added to openhgnn with the :func:`register_pipe`
    function decorator.

    For example::

        @register_task('modulation_classification')
        class ModulationClassification(BasePipe):
            (...)

    Args:
        name (str): the name of the pipes
    """

    def register_pipe_cls(cls):
        if name in PIPE_REGISTRY:
            raise ValueError("Cannot register duplicate pipe ({})".format(name))
        if not issubclass(cls, (BasePipe,ABC)):
            raise ValueError("Pipe ({}: {}) must extend BasePipe or ABC".format(name, cls.__name__))
        PIPE_REGISTRY[name] = cls
        return cls

    return register_pipe_cls


def try_import_pipe(pipe):
    if pipe not in PIPE_REGISTRY:
        if pipe in SUPPORTED_PIPES:
            importlib.import_module(SUPPORTED_PIPES[pipe])
        else:
            print(f"Failed to import {pipe} pipes.")
            return False
    return True


def build_pipe(args, pipe_name):
    if not try_import_pipe(pipe_name):
        exit(1)
    return PIPE_REGISTRY[pipe_name](args)