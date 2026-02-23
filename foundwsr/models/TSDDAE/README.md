# TS-DDAE: A Novel Temporal-Spectral Denoising Diffusion AutoEncoder for Wireless Signal Recognition Model Pre-training

This file contains the usage and description of the ICLR 2026 work TS-DDAE.

## How to use
**STEP1: Get the dataset**
We have not provide the dataset download function, so you may need to download the raw data. The dataset used in this work can be downloaded from:

- RML2016.10A (611 MB): https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom
- RML2016.10B (3.26 GB): https://www.kaggle.com/datasets/marwanabudeeb/rml201610b
- RML2018 (19.98 GB): https://www.kaggle.com/datasets/pinxau1000/radioml2018
- TechRec (1.76 GB): https://cloud.ilabt.imec.be/index.php/s/qrJCWgzQaGPfHPr
- ICARUS (4.76 GB):  https://genesys-lab.org/ICARUS

**STEP2: Run the training script**
You may run the following script for pretraining.
  ```bash
  python main.py -m TSDDAE -t pretrain -d RML2016.10a -g 0 --compile
  ```
Then, you can get the default checkpoint weight in "/OpenWSR/output/TSDDAE/TSDDAE_pretrain.pt". You can rename it to "/OpenWSR/output/TSDDAE/TSDDAE_RML2016.10a_amc.pt", and then use the following script for fine-tuning.
  ```bash
  python main.py -m TSDDAE -t amc -d RML2016.10a -g 0 --compile --load_from_pretrained
  ```
The hyper-parameters used is listed in "./config.yaml".

**Note**: If you find the STEP2 process too cumbersome, you can use the files in the "/example/ts_ddae.py" to directly run pre-training and fine-tuning without manually changing the names of the intermediate weight files.
