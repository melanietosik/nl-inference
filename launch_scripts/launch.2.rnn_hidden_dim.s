#!/bin/bash

#SBATCH --job-name=SNLI-RNN-hidden-dim
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mt3685@nyu.edu
#SBATCH --output=log.rnn_hidden_dim.txt

# Vary hidden dim
python run.py --model rnn --lr 1e-3 --hidden-dim 25 --id rnn_hidden_dim_25
python run.py --model rnn --lr 1e-3 --hidden-dim 50 --id rnn_hidden_dim_50
python run.py --model rnn --lr 1e-3 --hidden-dim 100 --id rnn_hidden_dim_100
python run.py --model rnn --lr 1e-3 --hidden-dim 250 --id rnn_hidden_dim_250
