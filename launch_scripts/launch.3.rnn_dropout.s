#!/bin/bash

#SBATCH --job-name=SNLI-RNN-dropout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mt3685@nyu.edu
#SBATCH --output=log.rnn_dropout.txt

# Vary dropout probability
python run.py --model rnn --hidden-dim 250 --dropout-prob 0.0 --id rnn_dropout_0_0
python run.py --model rnn --hidden-dim 250 --dropout-prob 0.2 --id rnn_dropout_0_2
python run.py --model rnn --hidden-dim 250 --dropout-prob 0.5 --id rnn_dropout_0_5
