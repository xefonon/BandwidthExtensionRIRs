#!/bin/sh
#BSUB -q hpc
#BSUB -J validation_data
#BSUB -n 1
#BSUB -W 64:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u xenoka@elektro.dtu.dk
#BSUB -o ../log/log-%J-%I.out
#BSUB -e ../log/log-%J-%I.err
# -- end of LSF options --

source /work3/xenoka/miniconda3/bin/activate pytorch

python process_validation_files.py

