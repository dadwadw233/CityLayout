#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=result.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

source activate layout

python src/train_lightning.py -cn=refactoring_sample 
