#!/bin/bash
#BSUB -J ex12
#BSUB -q gpua100
#BSUB -W 00:30
#BSUB -R "rusage[mem=6GB]"
#BSUB -n 8
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -o ../logs/ex12_%J.out
#BSUB -e ../errors/ex12_%J.err
#BSUB -B
#BSUB -N


source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613
python ../script_task12-using11.py
