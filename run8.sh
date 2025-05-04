#!/bin/sh
#BSUB -q c02613
#BSUB -J "run8_hpc"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model == XeonGold6126]"
#BSUB -W 00:10
#BSUB -R "rusage[mem=4GB]"
#BSUB -o jobs/run_hpc%J.out
#BSUB -e jobs/run_hpc%J.err

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613

#kernprof -l -v script_task8.py 20
python script_task8.py 20