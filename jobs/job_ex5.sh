#!/bin/bash
#BSUB -J ex5
#BSUB -q hpc
#BSUB -W 01:30
#BSUB -R "rusage[mem=50MB]"
#BSUB -n 20
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -o ../logs/ex5_%J.out
#BSUB -e ../errors/ex5_%J.err
#BSUB -B
#BSUB -N

lscpu

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613
time python3 ../ex5/simulate_ex5.py 20 1,2,4,8,12,16,20