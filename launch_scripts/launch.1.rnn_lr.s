#!/bin/bash

#SBATCH --job-name=SNLI-RNN-lr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mt3685@nyu.edu
#SBATCH --output=log.rnn_lr.txt

python run.py --model rnn --lr 1e-3 --id rnn_lr_1e_3
python run.py --model rnn --lr 5e-4 --id rnn_lr_5e_4
python run.py --model rnn --lr 1e-4 --id rnn_lr_1e_4
