# From Denoising to De-Channeling: Integrating Physical Channel Priors into Diffusion Models for Radio Signal Understanding

This file contains the usage and description of the ICML 2026 **Spotlight** work PWC-Diff.

## How to use
**STEP1: Get the dataset**
We have not provide the dataset download function, so you may need to download the raw data. The dataset used in this work can be downloaded from:

- RML2016.10A (611 MB): https://www.kaggle.com/datasets/nolasthitnotomorrow/radioml2016-deepsigcom
- RML2022 (451 MB): https://drive.google.com/drive/folders/1dEv6gPwPahUfFFRYYxvp3i5M34D7KI9J
- RML2018 (19.98 GB): https://www.kaggle.com/datasets/pinxau1000/radioml2018
- GNSS (1.9 GB): https://zenodo.org/records/4629685

**STEP2: Run the training script**
You may run the following script for our model.
  ```bash
  python main.py -m PWC_Diff -t pwcdiff -d RML2016.10a -g 0 --compile
  ```
The pre-training and fine-tuning are integrated into a single pipeline file.

The hyper-parameters used is listed in "./config.yaml".

