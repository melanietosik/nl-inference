#!/bin/bash

#SBATCH --job-name=SNLI-RNN
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mt3685@nyu.edu
#SBATCH --output=log.rnn.txt

python run.py --epochs 10 --num-workers 16 --hidden-dim 5000
