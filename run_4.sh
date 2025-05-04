#!/bin/sh
#BSUB -q hpc
#BSUB -J "kernprof"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -W 00:30
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "select[model==XeonGold6126]" #|| model==Xeon_Gold_6142 || model==Xeon_Gold_6226R]"  # CPU model constraint
#BSUB -B
#BSUB -N
#BSUB -o output/timingoutputxeongold6126_1%.txt    # Output file
#BSUB -e error/timingerrorxeongold6126_1%.txt     # Error file


source /dtu/projects/02613_2024/conda/conda_init.sh
conda activate 02613


#time python simulate.py 20

kernprof -l -v simulate.py 1
