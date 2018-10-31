#!/bin/bash

#SBATCH --job-name=SNLI-RNN-best
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mt3685@nyu.edu
#SBATCH --output=log.rnn_best.txt

python run.py --model rnn --epochs 10 --hidden-dim 250 --id rnn_best
