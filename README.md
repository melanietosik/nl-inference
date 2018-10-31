# nl-inference

CNN/RNN model evaluation on the "Stanford Natural Language Inference" (SNLI) task

## Task

- https://nlp.stanford.edu/projects/snli/
- https://nlp.stanford.edu/pubs/snli_paper.pdf

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

### MultiNLI

Use `split_mnli.py` to split the original `mnli_val.tsv` file into the corresponding subsets for each of the 5 genres. Then you can run the commands below to load and evaluate the best `CNN`/`RNN` model on the MultiNLI subsets as shown below.

```
$ # RNN
$ python run_mnli.py --model rnn --hidden-dim 250 --val /scratch/mt3685/nl_data/mnli_val.fiction.tsv
$ python run_mnli.py --model rnn --hidden-dim 250 --val /scratch/mt3685/nl_data/mnli_val.government.tsv
$ python run_mnli.py --model rnn --hidden-dim 250 --val /scratch/mt3685/nl_data/mnli_val.slate.tsv
$ python run_mnli.py --model rnn --hidden-dim 250 --val /scratch/mt3685/nl_data/mnli_val.telephone.tsv
$ python run_mnli.py --model rnn --hidden-dim 250 --val /scratch/mt3685/nl_data/mnli_val.travel.tsv
$ # CNN
$ python run_mnli.py --model cnn --hidden-dim 500 --kernel-size 2 --val /scratch/mt3685/nl_data/mnli_val.fiction.tsv
$ python run_mnli.py --model cnn --hidden-dim 500 --kernel-size 2 --val /scratch/mt3685/nl_data/mnli_val.government.tsv
$ python run_mnli.py --model cnn --hidden-dim 500 --kernel-size 2 --val /scratch/mt3685/nl_data/mnli_val.slate.tsv
$ python run_mnli.py --model cnn --hidden-dim 500 --kernel-size 2 --val /scratch/mt3685/nl_data/mnli_val.telephone.tsv
$ python run_mnli.py --model cnn --hidden-dim 500 --kernel-size 2 --val /scratch/mt3685/nl_data/mnli_val.travel.tsv
```

To "inspect" to best models and retrieve correct and incorrect predictions from the SNLI validation set, run:

```
$ # RNN
$ python run_mnli.py --model rnn --hidden-dim 250 --inspect 1
$ # CNN
$ python run_mnli.py --model cnn --hidden-dim 500 --kernel-size 2 --inspect 1
```

## Results

See `report.pdf` for a detailed write-up of the experimental results.

### SNLI

The best CNN model achieves **71.5** accuracy on the SNLI validation set and consists of 1303003 trained parameters  (cf.`cnn.pt.txt` and `log.cnn_best.txt`).

The best RNN model achieves **72.8** accuracy on the SNLI validation set and consists of 1079003 trained parameters  (cf.`rnn.pt.txt` and `log.rnn_best.txt`).

Note that the models were trained on only a subset of SNLI (approx. 100,000 training samples).
