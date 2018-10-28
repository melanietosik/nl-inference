import argparse

import torch
import torch.nn.functional as F

import constants as const
import data_loader

from RNN import RNN


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

    @return
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
        if (ix % args.log_interval == args.log_interval - 1):
            print(
                "epoch: [{:>2}/{:>2}]; step: [{:>3}/{:>3}]; loss: {:.6f}".format(
                    epoch, args.epochs, ix + 1, len(train_loader), loss))

    # Validation
    acc = validate_model(val_loader, model, device)
    print("\n epoch: [{:>2}/{:>2}]; validation accuracy: {}".format(
        epoch, args.epochs, acc))


def validate_model(val_loader, model, device):
    """
    Helper function to evaluate model performance on the validation set

    @param loader: validation DataLoader()
    @param model: CNN() or RNN()
    """
    correct = 0
    total = 0

    model.eval()
    for p, h, target in val_loader:
        p, h, target = p.to(device), h.to(device), target.to(device)

        output = F.softmax(model(p, h), dim=1)
        pred = output.max(1, keepdim=True)[1]

        total += target.size(0)
        correct += pred.eq(target.view_as(pred)).sum().item()

    return (100 * correct / total)


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
    model = RNN(
        vocab_size=const.MAX_VOCAB_SIZE,    # Vocabulary size
        emb_dim=const.EMB_DIM,              # Embedding dimensions
        hidden_dim=args.hidden_dim,         # Hidden dimensions
        padding_idx=const.PAD_IDX,          # Padding token index
        num_classes=const.NUM_CLASSES,      # Number of class labels
        id2tok=id2tok,                      # Vocabulary
    ).to(device)

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Main training loop
    for epoch in range(1, args.epochs + 1):

        print("\n{} epoch: {} {}".format("=" * 20, epoch, "=" * 20))

        train(args, model, device, train_loader, val_loader,
              optimizer, criterion, epoch)

    print("*" * 5 + "\n")
    return


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
    parser.add_argument("--log-interval", type=int, default=10, metavar="L",
                        help="training log interval (default: 10)")
    parser.add_argument("--use-cuda", type=int, default=1, metavar="C",
                        help="use CUDA (default: 1)")
    parser.add_argument("--seed", type=int, default=42, metavar="S",
                        help="random seed (default: 42)")

    parser.add_argument("--emb-dim", type=int, default=300, metavar="D",
                        help="embedding dimensions (default: 300)")
    parser.add_argument("--hidden-dim", type=int, default=100, metavar="H",
                        help="hidden dimensions (default: 100)")

    parser.add_argument("--train", type=str,
                        default="/scratch/mt3685/nl_data/snli_train.tsv",
                        metavar="T", help="training file path")
    parser.add_argument("--val", type=str,
                        default="/scratch/mt3685/nl_data/snli_val.tsv",
                        metavar="V", help="validation file path")

    args = parser.parse_args()
    main(args)
