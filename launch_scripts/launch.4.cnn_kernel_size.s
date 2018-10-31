#!/bin/bash

#SBATCH --job-name=SNLI-CNN-kernel-size
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=mt3685@nyu.edu
#SBATCH --output=log.cnn_kernel_size.txt

# Vary kernel size
python run.py --model cnn --hidden-dim 500 --kernel-size 1 --id cnn_kernel_size_1
python run.py --model cnn --hidden-dim 500 --kernel-size 2 --id cnn_kernel_size_2
python run.py --model cnn --hidden-dim 500 --kernel-size 3 --id cnn_kernel_size_3
