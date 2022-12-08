#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J csgm_inference
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
# request 16GB of system-memory
#BSUB -R "rusage[mem=8GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
#BSUB -o ../../log/log-%J-%I.out
#BSUB -e ../../log/log-%J-%I.err
# -- end of LSF options --

source /work3/xenoka/miniconda3/bin/activate pytorch

python run_CSGM.py --get_metrics --adaptive_gan