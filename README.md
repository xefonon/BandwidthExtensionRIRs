Generative models for sound field reconstruction
================================================

This repository contains the code for the paper [Generative models for sound field reconstruction](https://doi.org/10.1121/10.0016896) published in The Journal of the Acoustical Society of America (2023).

Contents
--------------------

    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── bin
    ├── config
    ├── data
    │   ├── Inference Data
    │   ├── PlaneWaveData
    │   └── WaveGAN Data
    ├── docs
    └── src
        ├── models
        │   ├── CSGM
        │   ├── HiFiGAN
        │   ├── SEGAN
        │   └── WaveGAN
        ├── tools
        └── visualization

Usage
--------------------

To create a conda environment with the required dependencies, run the following command in your terminal:

`conda env create -f environment.yml`

To train the models, you must first generate the synthetic data.
This can be done by running the following command in the terminal:

`python ./data/PlaneWaveData/PlaneWaveArrays.py --lsf_index $n --init_n_mics 100 --radius 0.5`


where `$n` is the index which corresponds to [0 - N<sub>max</sub>] sound fields. 
Each of these are a random wave field impinging on a spherical microphone array with 
the number of elements set by `--init_n_mics` and radius set by `--radius`.

For the CSGM model, one must first train the WaveGAN model. The data for the WaveGAN model
is obtained by running the terminal command:

`python data/WaveGAN Data/WaveGAN_Datasets.py`.

To train the WaveGAN for the CSGM optimisation run the following command:


` python ./models/WaveGAN/train_wavegan.py --config_file './config/wavegan_config.yaml'`

where the `--config_file` argument is the path to the config file.

To train the HiFiGAN model run the following command:

` python /models/HiFiGAN/train.py --config_file '/config/HiFiGAN_config.yaml'`


To train the SEGAN model run the following command:

` python /models/SEGAN/train.py --config_file '/config/SEGAN_config.yaml'`

An example of each network's performance is given in the notebook
./src/visualization/bandwidth_extension_example.ipynb. To run this notebook,
first run the script ./src/visualization/download_example_data.py to download the necessary
files.


Citation
--------------------

If you use this code, please cite the paper as:

```@article{EFG_GANRIR,
author = {Fernandez-Grande,Efren  and Karakonstantis,Xenofon  and Caviedes-Nozal,Diego  and Gerstoft,Peter },
title = {Generative models for sound field reconstruction},
journal = {The Journal of the Acoustical Society of America},
volume = {153},
number = {2},
pages = {1179-1190},
year = {2023},
doi = {10.1121/10.0016896}}
