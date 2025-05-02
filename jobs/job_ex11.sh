#!/bin/bash
#BSUB -J ex11
#BSUB -q c02613
#BSUB -W 00:10
#BSUB -R "rusage[mem=4GB]"
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -o ../logs/ex11_%J.out
#BSUB -e ../errors/ex11_%J.err
#BSUB -B
#BSUB -N


source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613

nsys profile -o ../logs/ex11_profile python ../script_task11.py
