# README

This software repository contains the data and script files necessary to reproduce the results of Van den Ende et al.: _A Self-Supervised Deep Learning Approach for Blind Denoising and Waveform Coherence Enhancement in Distributed Acoustic Sensing data_. The Jupyter notebooks are organised as follows:

- `prepare_synthetics.ipynb`: Downloads the waveform data from the SCSN/IRIS data centre and generates the synthetic strain rate data.
- `train_j-invariant_synthetic.ipynb`: Trains a J-invariant model on synthetic data.
- `train_j-invariant_DAS.ipynb`: Trains a J-invariant model on DAS data, using a model pretrained on synthetics.
- `Figs1-3_zebra_architecture_cable-location.ipynb`: Generates the non-data figures 1 to 3.
- `Figs4-5_synthetic-results.ipynb`: Generates figures 4 and 5 on the synthetic data results.
- `Figs6-9_DAS-results.ipynb`: Generates figures 6 to 9 on the DAS data results.

The models trained on synthetics (`pretrained_synthetic.h5`) and on DAS data (`pretrained_DAS.h5`) are included in the `save` directory. The data required to reproduce the figures are included in the `data` directory, and so it is not required to run `prepare_synthetics.ipynb`, `train_j-invariant_synthetic.ipynb`, and `train_j-invariant_DAS.ipynb`. Python dependencies are included in `requirements.txt`.
