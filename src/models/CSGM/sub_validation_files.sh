#!/bin/sh
#BSUB -q hpc
#BSUB -J validation_files[8001-8193]
#BSUB -n 1
#BSUB -W 10:00
#BSUB -R "rusage[mem=6GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o ../log/log-%J-%I.out
#BSUB -e ../log/log-%J-%I.err
# -- end of LSF options --

source /work3/xenoka/miniconda3/bin/activate pytorch

python process_validation_files.py --decimate True --rec_indx $LSB_JOBINDEX
