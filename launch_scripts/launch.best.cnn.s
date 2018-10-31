#!/bin/bash

#SBATCH --job-name=SNLI-CNN-best
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mt3685@nyu.edu
#SBATCH --output=log.cnn_best.txt

python run.py --model cnn --epochs 10 --lr 1e-3 --hidden-dim 500 --kernel-size 2 --id cnn_best
