#!/bin/sh
#BSUB -q hpc
#BSUB -J "run9"
#BSUB -n 1
#BSUB -R "select[model == XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -W 00:10
#BSUB -R "rusage[mem=4GB]"

if [ ! -d "jobs" ]; then
    mkdir jobs
fi
#BSUB -o jobs/run_9%J.out
#BSUB -e jobs/run_9%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613


kernprof script_task9.py 20
#
#python script_task9.py 20
