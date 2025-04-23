#!/bin/bash
#BSUB -J ex2
#BSUB -q hpc
#BSUB -W 00:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -o ../logs/ex2_%J.out
#BSUB -e ../errors/ex2_%J.err
#BSUB -B
#BSUB -N

# lscpu

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613
time python3 ../simulate.py 20