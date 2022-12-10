RIR Bandwidth Extension
==============================

This repository contains the code for the paper
"Generative adversarial models for reconstructing room impulse responses" by
E. Fernandez-Grande, X. Karakonstantis, D. Caviedes Nozal, and P. Gerstoft submitted to
the Journal of the Acoustical Society of America (JASA). 

The paper will be available in the near future.

Project Organization
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

To train the models, you must first generate the synthetic data. 
This can be done by running the terminal command

`python ./data/PlaneWaveData/PlaneWaveArrays.py --lsf_index $n --init_n_mics 100 --radius 0.5`


where `$n` is the index which corresponds to [0 - N<sub>max</sub>] sound fields. 
Each of these are a random wave field impinging on a spherical microphone array with 
the number of elements set by `--init_n_mics` and radius set by `--radius`.

For the CSGM model, one must first train the WaveGAN model. The data for the WaveGAN model
is obtained by running the terminal command 

`python data/WaveGAN Data/WaveGAN_Datasets.py`.

To train the WaveGAN for the CSGM optimisation run the following command:


` python ./models/WaveGAN/train_wavegan.py --config_file './config/wavegan_config.yaml'`

where the `--config_file` argument is the path to the config file.

To train the HiFiGAN model run the following command:

` python /models/HiFiGAN/train.py --config_file '/config/HiFiGAN_config.yaml'`


To train the SEGAN model run the following command:

` python /models/SEGAN/train.py --config_file '/config/SEGAN_config.yaml'`

An example of each networks' performance is given in the notebook 
`./src/visualization/bandwidth_extension_example.ipynb`. To run this notebook,
first run the script `./src/visualization/download_example_data.py` to download the necessary
files.