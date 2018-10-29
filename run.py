import argparse
import json
import numpy as np
import torch

import constants as const
import data_loader

from RNN import RNN
from CNN import CNN


def train(args, model, device, train_loader, val_loader,
          optimizer, criterion, epoch):
    """
    Training function

    @param args: command line arguments
    @param model: CNN() or RNN()
    @param device: device type ("cpu" or "cuda")
    @param train_loader: training DataLoader()
    @param val_loader: validation DataLoader()
    @param optimizer: optimizer object
    @param optimizer: criterion object
    @param epoch: current epoch
    """
    model.train()

    # Iterate over mini-batches
    for ix, (p, h, target) in enumerate(train_loader):

        p, h, target = p.to(device), h.to(device), target.to(device)
        optimizer.zero_grad()

        # Forward pass
        output = model(p, h)
        loss = criterion(output, target)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Logging
        if ((ix > 0) and (ix % args.log_interval == 0)):

            model.eval()

            # Logging
            train_acc, train_loss = eval_model(
                train_loader, model, device, criterion)
            val_acc, val_loss = eval_model(
                val_loader, model, device, criterion)
            logging["train_accs"].append(train_acc)
            logging["train_loss"].append(train_loss)
            logging["val_accs"].append(val_acc)
            logging["val_loss"].append(val_loss)

            print(
                "epoch: [{:>2}/{:>2}]; step: [{:>3}/{:>3}]; loss: {:.6f}".format(
                    epoch, args.epochs, ix + 1, len(train_loader), loss))

            model.train()

    # Final validation
    val_acc, val_loss = eval_model(val_loader, model, device)
    print("\n epoch: [{:>2}/{:>2}]; val acc: {}; val loss: {}".format(
        epoch, args.epochs, val_acc, val_loss))


def eval_model(loader, model, device, criterion):
    """
    Helper function to evaluate model performance on the given dataset

    @param loader: DataLoader()
    @param model: CNN() or RNN()
    @param device: device
    @
    """
    correct = 0
    total = 0
    running_loss = 0.0  # Running sum of batch loss value

    for p, h, target in loader:

        assert(len(p) == len(h) == len(target))
        p, h, target = p.to(device), h.to(device), target.to(device)

        output = model(p, h)
        pred = output.max(1, keepdim=True)[1]

        loss = criterion(output, target)
        running_loss += loss.item() * len(target)  # Undo "elementwise_mean"

        total += target.size(0)
        correct += pred.eq(target.view_as(pred)).sum().item()

    return (100 * correct / total), (running_loss / total)


def main(args):
    """
    Main function
    """
    # Use CUDA
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device: \"{}\"".format(device))

    # Fix random seed
    torch.manual_seed(args.seed)

    # Generate token-to-index and index-to-token mapping
    tok2id, id2tok = data_loader.build_or_load_vocab(
        args.train, overwrite=True)

    print("*" * 5)
    print(args)

    # Create DataLoader() objects
    params = {
        "batch_size": args.batch_size,
        "collate_fn": data_loader.collate_fn,
        "shuffle": args.shuffle,
        "num_workers": args.num_workers,
    }
    train_dataset = data_loader.SNLIDataSet(args.train, tok2id)
    train_loader = torch.utils.data.DataLoader(train_dataset, **params)
    val_dataset = data_loader.SNLIDataSet(args.val, tok2id)
    val_loader = torch.utils.data.DataLoader(val_dataset, **params)

    # Initialize model
    if args.model == "rnn":  # RNN model
        model = RNN(
            vocab_size=const.MAX_VOCAB_SIZE,    # Vocabulary size
            emb_dim=const.EMB_DIM,              # Embedding dimensions
            hidden_dim=args.hidden_dim,         # Hidden dimensions
            dropout_prob=args.dropout_prob,     # Dropout probability
            padding_idx=const.PAD_IDX,          # Padding token index
            num_classes=const.NUM_CLASSES,      # Number of class labels
            id2tok=id2tok,                      # Vocabulary
        ).to(device)
    elif args.model == "cnn":  # CNN model
        model = CNN(
            vocab_size=const.MAX_VOCAB_SIZE,    # Vocabulary size
            emb_dim=const.EMB_DIM,              # Embedding dimensions
            hidden_dim=args.hidden_dim,         # Hidden dimensions
            kernel_size=args.kernel_size,       # Kernel size
            dropout_prob=args.dropout_prob,     # Dropout probability
            padding_idx=const.PAD_IDX,          # Padding token index
            num_classes=const.NUM_CLASSES,      # Number of class labels
            id2tok=id2tok,                      # Vocabulary
        ).to(device)
    else:
        print("Invalid model specification, exiting")
        exit()

    # Criterion
    criterion = torch.nn.CrossEntropyLoss()
    # Model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum([np.prod(p.size()) for p in params])
    # Optimizer
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # Logging
    global logging
    logging = {
        "train_accs": [],
        "train_loss": [],
        "val_accs": [],
        "val_loss": [],
        "num_params": num_params,
    }

    # Main training loop
    for epoch in range(1, args.epochs + 1):
        # Log epoch
        print("\n{} epoch: {} {}".format("=" * 20, epoch, "=" * 20))
        # Train model
        train(args, model, device, train_loader, val_loader,
              optimizer, criterion, epoch)

    print("*" * 5 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch NL inference")

    parser.add_argument("--batch-size", type=int, default=256, metavar="N",
                        help="mini-batch size for training (default: 256)")
    parser.add_argument("--num-workers", type=int, default=8, metavar="N",
                        help="number of worker threads (default: 8)")
    parser.add_argument("--shuffle", type=int, default=1, metavar="S",
                        help="shuffle training data (default: 1)")

    parser.add_argument("--epochs", type=int, default=1, metavar="E",
                        help="number of epochs to train (default: 1)")
    parser.add_argument("--log-interval", type=int, default=100, metavar="L",
                        help="training log interval (default: 100)")
    parser.add_argument("--use-cuda", type=int, default=1, metavar="C",
                        help="use CUDA (default: 1)")
    parser.add_argument("--seed", type=int, default=42, metavar="S",
                        help="random seed (default: 42)")

    parser.add_argument("--emb-dim", type=int, default=300, metavar="D",
                        help="embedding dimensions (default: 300)")
    parser.add_argument("--hidden-dim", type=int, default=100, metavar="H",
                        help="hidden dimensions (default: 100)")
    parser.add_argument("--kernel-size", type=int, default=3, metavar="K",
                        help="kernel size (default: 3)")
    parser.add_argument("--dropout-prob", type=float, default=0.0, metavar="D",
                        help="dropout probability (default: 0.0)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="L",
                        help="learning rate (default: 0.001)")

    parser.add_argument("--model", type=str, default="rnn", metavar="M",
                        help="neural network model")
    parser.add_argument("--train", type=str,
                        default="/scratch/mt3685/nl_data/snli_train.tsv",
                        metavar="T", help="training file path")
    parser.add_argument("--val", type=str,
                        default="/scratch/mt3685/nl_data/snli_val.tsv",
                        metavar="V", help="validation file path")
    parser.add_argument("--id", type=str, default="debug", metavar="I",
                        help="experiment ID")

    args = parser.parse_args()
    main(args)

    # Dump logging metrics to file
    with open("logging.id={}.json".format(args.id), "w") as fp:
        json.dump(logging, fp)
