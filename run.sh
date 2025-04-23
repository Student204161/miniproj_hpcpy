#!/bin/sh
#BSUB -q hpc
#BSUB -J "run"
#BSUB -n 1
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -W 00:05
#BSUB -R "rusage[mem=4GB]"
#BSUB -u s204161@dtu.dk
#BSUB -B
#BSUB -N
if [ ! -d "jobs" ]; then
    mkdir jobs
fi
#BSUB -o jobs/run_%J.out
#BSUB -e jobs/run_%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613



python simulate.py 20 