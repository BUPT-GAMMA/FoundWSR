# FoundWSR

This is a repository for building Wireless Signal Recognition (WSR) foundation models based on PyTorch, integrating various signal datasets, models, and model experiment pipelines. We try to provide a user-transparent and easy-to-use benchmarking interface.

**Note**: This version is currently only used as an implementation reference for TS-DDAE [ICLR 2026] and PWC-Diff [ICML 2026]. Updates to this repository will be made gradually in the future.

# Requirements and Installation

- PyTorch >= 2.2.0
- sklearn
- scipy
- pandas
- opencv-python-headless
- Optuna (if you want hyperparameters optimization)
- matplotlib, seaborn (if you want visualization)

# Get started

## Running an existing baseline model

```bash
python main.py -m model_name -d dataset_name -t task_name -g 0 --load_from_pretrained
```

*optional arguments*:

``--model -m ``    name of models

``--task -t``    name of task

``--dataset -d``    name of datasets

``--gpu -g``    controls which gpu you will use. If you do not have gpu, set -g -1.

``--load_from_pretrained`` will load the model from a default checkpoint.

``--compile`` will use the ``torch.compile`` optimization. (still under development)

Note: some usage examples are provided in "/example/". 

## Model config

We provide two ways of modifying the hyperparameters of the model

1. By specifying parameters in the experiment function in ``main.py``.
2. By modifying the ``config.yaml`` file in each model folder in the foundwsr/models.

**Note: the first method has higher priority than the second.**

## Pipeline support

If you want to add new pipelines, you need to

1. Inherit the BasePipe from ``foundwsr.pipeline.base_pipe``.

2. Register the pipeline name using ``register_pipe`` in ``foundwsr.pipeline.__init__``.

3. Add the pipeline name and file path to the ``SUPPORTED_PIPES`` variable in ``foundwsr.pipeline.__init__``.

4. Write the pipeline's operation logic.

5. Add the task name and pipeline name to ``supported.yaml`` in the foundwsr folder.

The pipeline calling logic has been encapsulated in the ``experiment.py`` file in the ``foundwsr`` folder.

## Dataset support

Similar to the pipeline support, if you want to add new datasets, you need to

1. Inherit the BaseDataset from ``foundwsr.dataset.base_dataset``.

2. Register the dataset name using ``register_dataset`` in ``foundwsr.dataset.__init__``.

3. Add the dataset name and file path to the ``SUPPORTED_DATASETS`` variable in ``foundwsr.dataset.__init__``.

4. Write the dataset's operation logic.

Note: we recommend you to write the @classmethod ``create`` if you want to enjoy the ``build_dataset`` from ``foundwsr.dataset.__init__``. In ``create``, users can define the initial data loading and data partitioning. For task-specific dataset like ``MaskedDataset``, you may not write the ``create``.

## Model support

Similar to the dataset support, if you want to add new models, you need to

1. Create a folder in the ``foundwsr.models``.

2. Inherit the BaseModel from ``foundwsr.models.base_model``.

3. Register the model name using ``register_model`` in ``foundwsr.models.__init__``.

4. Add the model name and file path to the ``SUPPORTED_MODELS`` variable in ``foundwsr.models.__init__``.

5. Write the model's operation logic.

6. Add ``config.yaml`` to specify the hyperparameters used in this model.

Note: we recommend you to write the @classmethod ``build_model_from_args`` if you want to enjoy the ``build_model`` from ``foundwsr.model.__init__``. In ``build_model_from_args``, users can define the initial input hyperparameters.

**This repository is still under development. If you have any questions, suggestions, or would like to contribute, please open an issue, submit a pull request, or contact me directly at yaoqiliu@bupt.edu.cn**