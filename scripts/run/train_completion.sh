#!/bin/bash
#SBATCH --job-name=citylayout_train
#SBATCH --output=result.txt
#SBATCH --nodes=1

#SBATCH --gres=gpu:4

source activate layout

python src/train_lightning.py -cn=refactoring_train_completion
