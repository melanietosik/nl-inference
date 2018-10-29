# nl-inference

CNN/RNN model evaluation on the Stanford Natural Language Inference (SNLI) task

## Requirements

```
$ module load anaconda3/5.3.0  # HPC only
$ module load cuda/9.0.176 cudnn/9.0v7.0.5  # HPC only
$ conda create -n nli python=3.6
$ conda activate nli
$ conda install torch pandas numpy
```

On HPC, you might need to add the following line to your `~/.bashrc`:

`. /share/apps/anaconda3/5.3.0/etc/profile.d/conda.sh`

## Run

All the scripts to launch the individual experiments are stored in the `launch_scripts/` directory. You can submit them using `$ sbatch launch.xyz.s`. The scripts will run on a standard GPU node by default.

## Results

See `report.pdf` for a detailed write-up of the experimental results.
