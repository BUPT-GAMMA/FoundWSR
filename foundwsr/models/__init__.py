import importlib
from .base_model import BaseModel
from torch import nn
import sys

sys.path.append("..")

MODEL_REGISTRY = {}
SUPPORTED_MODELS = {
    "IQFormer": "foundwsr.models.IQFormer.IQFormer",
    "MoCo_IQFormer": "foundwsr.models.MoCo_IQFormer.MoCo_IQFormer",
    "RF_Diffusion": "foundwsr.models.RF_Diffusion.RF_Diffusion",
    "SpectrumFM": "foundwsr.models.SpectrumFM.SpectrumFM",
    "AMC_Net": "foundwsr.models.AMC_Net.AMC_Net",
    "CGDNN": "foundwsr.models.CGDNN.CGDNN",
    "CNN2": "foundwsr.models.CNN2.CNN2",
    "DAE": "foundwsr.models.DAE.DAE",
    "GRU2": "foundwsr.models.GRU2.GRU2",
    "MCNet": "foundwsr.models.MCNet.MCNet",
    "MSNet": "foundwsr.models.MSNet.MSNet",
    "ResNet": "foundwsr.models.ResNet.ResNet",
    "Transformer": "foundwsr.models.Transformer.Transformer",
    "VGG": "foundwsr.models.VGG.VGG",
    "LDM_Encoder": "foundwsr.models.LDM.LDM_Encoder",
    "LDM": "foundwsr.models.LDM.LDM",
    "DDAE_Network": "foundwsr.models.DDAE.Network",
    "custom_encoder": "foundwsr.models.custom.custom_encoder",
    "custom": "foundwsr.models.custom.custom",
    "TSDDAE": "foundwsr.models.TSDDAE.TSDDAE",
    "PWCDiff": "foundwsr.models.PWCDiff.PWCDiff",
    "signal": "foundwsr.models.signal.signal"
}

def register_model(name):
    """
    New models types can be added with the :func:`register_model`
    function decorator.

    For example::

        @register_model('gat')
        class GAT(BaseModel):
            (...)

    Args:
        name (str): the name of the models
    """

    def register_model_cls(cls):
        if name in MODEL_REGISTRY:
            raise ValueError("Cannot register duplicate models ({})".format(name))
        if not issubclass(cls, BaseModel):
            raise ValueError(
                "Model ({}: {}) must extend BaseModel".format(name, cls.__name__)
            )
        MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls

    return register_model_cls


def try_import_model(model):
    if model not in MODEL_REGISTRY:
        if model in SUPPORTED_MODELS:
            importlib.import_module(SUPPORTED_MODELS[model])
        else:
            print(f"Failed to import {model} models.")
            return False
    return True


def build_model(model):
    if isinstance(model, nn.Module):
        if not hasattr(model, "build_model_from_args"):
            def build_model_from_args(args):
                return model
            model.build_model_from_args = build_model_from_args
        return model
    if not try_import_model(model):
        exit(1)
    return MODEL_REGISTRY[model]