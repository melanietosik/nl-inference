#!/bin/bash

#SBATCH --job-name=SNLI-CNN-dropout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mt3685@nyu.edu
#SBATCH --output=log.cnn_dropout.txt

# Vary dropout probability
python run.py --model cnn --hidden-dim 500 --dropout-prob 0.0 --id cnn_dropout_0_0
python run.py --model cnn --hidden-dim 500 --dropout-prob 0.2 --id cnn_dropout_0_2
python run.py --model cnn --hidden-dim 500 --dropout-prob 0.5 --id cnn_dropout_0_5
