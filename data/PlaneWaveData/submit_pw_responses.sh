#!/bin/sh
#BSUB -q hpc
#BSUB -J PWdata[1-30]
#BSUB -n 1
#BSUB -W 16:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -R "span[hosts=1]"
#BSUB -u xenoka@elektro.dtu.dk
#BSUB -o ../log/log-%J-%I.out
#BSUB -e ../log/log-%J-%I.err
# -- end of LSF options --

source /work3/xenoka/miniconda3/bin/activate pytorch

python3 PlaneWaveArrays.py --lsf_index $LSB_JOBINDEX --init_n_mics 100 --radius 0.5
