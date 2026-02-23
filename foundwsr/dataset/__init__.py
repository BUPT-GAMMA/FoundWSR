import importlib

DATASET_REGISTRY = {}
SUPPORTED_DATASETS = {
    "RML2016.10a": "foundwsr.dataset.rml2016",
    "RML2016.10b": "foundwsr.dataset.rml2016",
    "RML2016.04c": "foundwsr.dataset.rml2016",
    "RML2018": "foundwsr.dataset.rml2018",
    "RML2022": "foundwsr.dataset.rml2022",
    "HisarMod2019": "foundwsr.dataset.hisarmod2019",
    "Techrec": "foundwsr.dataset.techrec",
    "ICARUS": "foundwsr.dataset.icarus",
    "GNSS": "foundwsr.dataset.gnss",
    "RadChar": "foundwsr.dataset.radchar"
}

def register_dataset(name):
    """
    New dataset types can be added with the :func:`register_dataset`
    function decorator.

    For example::

        @register_dataset('my_dataset')
        class MyDataset():
            (...)

    Args:
        name (str): the name of the dataset
    """

    def register_dataset_cls(cls):
        if name in DATASET_REGISTRY:
            raise ValueError("Cannot register duplicate dataset ({})".format(name))
        DATASET_REGISTRY[name] = cls
        return cls

    return register_dataset_cls


def try_import_dataset(name):
    if name not in DATASET_REGISTRY:    
        if name in SUPPORTED_DATASETS:
            importlib.import_module(SUPPORTED_DATASETS[name])
        else:
            print(f"Failed to import {name} dataset.")
            return False
    return True

def build_dataset(dataset, dataset_path=None, *args, **kwargs):
    if not try_import_dataset(dataset):
        exit(1)
    if "RML2016" in dataset:
        DATASET_REGISTRY["RML2016"].create(dataset_path, dataset, *args, **kwargs)
        return DATASET_REGISTRY["RML2016"]
    else:
        DATASET_REGISTRY[dataset].create(dataset_path, *args, **kwargs)
        return DATASET_REGISTRY[dataset]
