#!/bin/bash
#BSUB -J ex6_uno50
#BSUB -q hpc
#BSUB -W 04:30
#BSUB -R "rusage[mem=50MB]"
#BSUB -n 20
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -o ../logs/ex6_%J.out
#BSUB -e ../errors/ex6_%J.err


lscpu

source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

time python3 ../ex6/simulate_ex6.py 50 1,2,4,8,12,16,20