import argparse
import numpy as np
import torch

import constants as const
import data_loader

from RNN import RNN
from CNN import CNN


def eval_model(loader, model, device, criterion, inspect=False):
    """
    Helper function to evaluate model performance on the given dataset

    @param loader: DataLoader()
    @param model: CNN() or RNN()
    @param device: device
    @param criterion: loss criterion
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

        if inspect:
            right_preds = []
            wrong_preds = []
            right_batched = pred.eq(target.view_as(pred))
            # Correct
            for i, right in enumerate(right_batched):
                right = right.item()
                if right:
                    right_preds.append(p[i].cpu().numpy())
                if len(right_preds) == 3:
                    break
            # Incorrect
            for i, right in enumerate(right_batched):
                right = right.item()
                if right:
                    continue
                wrong_preds.append(p[i].cpu().numpy())
                if len(wrong_preds) == 3:
                    break
            return right_preds, wrong_preds

    return (100 * correct / total), (running_loss / total)


def main(args):
    """
    Evaluate SNLI model on MNLI data set
    """
    # Use CUDA
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Fix random seed
    torch.manual_seed(args.seed)

    # Generate token-to-index and index-to-token mapping
    tok2id, id2tok = data_loader.build_or_load_vocab(
        args.train, overwrite=False)

    print("*" * 5)
    print(args)

    # Create DataLoader() objects
    params = {
        "batch_size": args.batch_size,
        "collate_fn": data_loader.collate_fn,
        "shuffle": args.shuffle,
        "num_workers": args.num_workers,
    }
    # train_dataset = data_loader.SNLIDataSet(args.train, tok2id)
    # train_loader = torch.utils.data.DataLoader(train_dataset, **params)
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
        # Load model weights from disk
        model.load_state_dict(torch.load(const.MODELS + "rnn.pt"))
        model.eval()
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
        # Load model weights from disk
        model.load_state_dict(torch.load(const.MODELS + "cnn.pt"))
        model.eval()
    else:
        print("Invalid model specification, exiting")
        exit()

    # Criterion
    criterion = torch.nn.CrossEntropyLoss()
    # Model parameters
    params = [p for p in model.parameters() if p.requires_grad]

    # Inspect correct/incorrect predictions
    if args.inspect:
        right, wrong = eval_model(val_loader, model, device, criterion,
                                  inspect=True)
        print("\nValidation premises with correct predictions:\n")
        for i, item in enumerate(right):
            text = " ".join([id2tok[idx] for idx in item if idx > 0])
            print("#{}\n {}".format(i + 1, text))
        print("\nValidation premises with incorrect predictions:\n")
        for i, item in enumerate(wrong):
            text = " ".join([id2tok[idx] for idx in item if idx > 0])
            print("#{}\n {}".format(i + 1, text))
        return

    # Validation
    val_acc, _ = eval_model(val_loader, model, device, criterion)
    print("\n Validation accuracy: {}".format(val_acc))

    print("*" * 5 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch NL inference")

    parser.add_argument("--batch-size", type=int, default=256, metavar="N",
                        help="mini-batch size for training (default: 256)")
    parser.add_argument("--num-workers", type=int, default=8, metavar="N",
                        help="number of worker threads (default: 8)")
    parser.add_argument("--shuffle", type=int, default=1, metavar="S",
                        help="shuffle training data (default: 1)")

    parser.add_argument("--epochs", type=int, default=5, metavar="E",
                        help="number of epochs to train (default: 5)")
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
    parser.add_argument("--lr", type=float, default=1e-3, metavar="L",
                        help="learning rate (default: 1e-3)")

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
    parser.add_argument("--inspect", type=int, default=0, metavar="I",
                        help="inspect correct/incorrect predictions")

    args = parser.parse_args()
    main(args)
