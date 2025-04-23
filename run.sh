#!/bin/sh
#BSUB -q hpc
#BSUB -J "run"
#BSUB -n 1
#BSUB -R "span[hosts=1]"
#BSUB -W 00:02
#BSUB -R "rusage[mem=4GB]"
#BSUB -u s204161@dtu.dk
#BSUB -B
#BSUB -N
#BSUB -o 1_6_%J.out
#BSUB -e 1_6_%J.err

/bin/sleep 60