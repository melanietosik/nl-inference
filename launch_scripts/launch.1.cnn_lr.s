#!/bin/bash

#SBATCH --job-name=SNLI-CNN-lr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mt3685@nyu.edu
#SBATCH --output=log.cnn_lr.txt

python run.py --model cnn --lr 1e-3 --id cnn_lr_1e_3
python run.py --model cnn --lr 5e-4 --id cnn_lr_5e_4
python run.py --model cnn --lr 1e-4 --id cnn_lr_1e_4
