import numpy as np
import json

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("lines", linewidth=1)
CB = ["#377eb8", "#ff7f00", "#4daf4a",
      "#f781bf", "#a65628", "#984ea3",
      "#999999", "#e41a1c", "#dede00"]

epochs = 5
lin = np.linspace(0, epochs, epochs * 3)


def plot_lr():
    """
    Learning rate
    """
    cnn_2 = "logging.cnn_lr_1e_3.json"
    cnn_3 = "logging.cnn_lr_5e_4.json"
    cnn_4 = "logging.cnn_lr_1e_4.json"
    rnn_2 = "logging.rnn_lr_1e_3.json"
    rnn_3 = "logging.rnn_lr_5e_4.json"
    rnn_4 = "logging.rnn_lr_1e_4.json"

    data = {}
    for exp in (cnn_2, cnn_3, cnn_4, rnn_2, rnn_3, rnn_4):
        data[exp.split(".")[1]] = json.load(open(exp, "r"))

    # CNN
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(30, 80)
    ax.set_xticks(np.arange(0, epochs + 1, 1))
    ax.set_title("Adam learning rate (CNN)")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")

    ax.plot(lin, data["cnn_lr_1e_3"]["train_accs"], color=CB[0],
            label="[1e-3] train")
    ax.plot(lin, data["cnn_lr_1e_3"]["val_accs"], color=CB[0], linestyle="--",
            label="[1e-3] val")
    ax.plot(lin, data["cnn_lr_5e_4"]["train_accs"], color=CB[1],
            label="[5e-4] train")
    ax.plot(lin, data["cnn_lr_5e_4"]["val_accs"], color=CB[1], linestyle="--",
            label="[5e-4] val")
    ax.plot(lin, data["cnn_lr_1e_4"]["train_accs"], color=CB[2],
            label="[1e-4] train")
    ax.plot(lin, data["cnn_lr_1e_4"]["val_accs"], color=CB[2], linestyle="--",
            label="[1e-4] val")
    ax.legend()
    plt.savefig("../plots/cnn_lr.eps", format="eps", dpi=500)
    plt.close()

    # RNN
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(30, 80)
    ax.set_xticks(np.arange(0, epochs + 1, 1))
    ax.set_title("Adam learning rate (RNN)")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")

    ax.plot(lin, data["rnn_lr_1e_3"]["train_accs"], color=CB[0],
            label="[1e-3] train")
    ax.plot(lin, data["rnn_lr_1e_3"]["val_accs"], color=CB[0], linestyle="--",
            label="[1e-3] val")
    ax.plot(lin, data["rnn_lr_5e_4"]["train_accs"], color=CB[1],
            label="[5e-4] train")
    ax.plot(lin, data["rnn_lr_5e_4"]["val_accs"], color=CB[1], linestyle="--",
            label="[5e-4] val")
    ax.plot(lin, data["rnn_lr_1e_4"]["train_accs"], color=CB[2],
            label="[1e-4] train")
    ax.plot(lin, data["rnn_lr_1e_4"]["val_accs"], color=CB[2], linestyle="--",
            label="[1e-4] val")
    ax.legend(loc="upper left")
    plt.savefig("../plots/rnn_lr.eps", format="eps", dpi=500)
    plt.close()


def plot_hidden_dim():
    """
    Hidden dimensions
    """
    cnn_50 = "logging.cnn_hidden_dim_50.json"
    cnn_100 = "logging.cnn_hidden_dim_100.json"
    cnn_200 = "logging.cnn_hidden_dim_200.json"
    cnn_500 = "logging.cnn_hidden_dim_500.json"
    rnn_25 = "logging.rnn_hidden_dim_25.json"
    rnn_50 = "logging.rnn_hidden_dim_50.json"
    rnn_100 = "logging.rnn_hidden_dim_100.json"
    rnn_250 = "logging.rnn_hidden_dim_250.json"

    data = {}
    for exp in (cnn_50, cnn_100, cnn_200, cnn_500,
                rnn_25, rnn_50, rnn_100, rnn_250):
        data[exp.split(".")[1]] = json.load(open(exp, "r"))

    # CNN
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(40, 90)
    ax.set_xticks(np.arange(0, epochs + 1, 1))
    ax.set_title("Hidden dimensions (CNN)")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")

    ax.plot(lin, data["cnn_hidden_dim_50"]["train_accs"], color=CB[0],
            label="[50] train")
    ax.plot(lin, data["cnn_hidden_dim_50"]["val_accs"], color=CB[0], linestyle="--",
            label="[50] val")
    ax.plot(lin, data["cnn_hidden_dim_100"]["train_accs"], color=CB[1],
            label="[100] train")
    ax.plot(lin, data["cnn_hidden_dim_100"]["val_accs"], color=CB[1], linestyle="--",
            label="[100] val")
    ax.plot(lin, data["cnn_hidden_dim_200"]["train_accs"], color=CB[2],
            label="[200] train")
    ax.plot(lin, data["cnn_hidden_dim_200"]["val_accs"], color=CB[2], linestyle="--",
            label="[200] val")
    ax.plot(lin, data["cnn_hidden_dim_500"]["train_accs"], color=CB[3],
            label="[500] train")
    ax.plot(lin, data["cnn_hidden_dim_500"]["val_accs"], color=CB[3], linestyle="--",
            label="[500] val")
    ax.legend(loc="upper left")
    plt.savefig("../plots/cnn_hidden_dims.eps", format="eps", dpi=500)
    plt.close()

    # RNN
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(40, 90)
    ax.set_xticks(np.arange(0, epochs + 1, 1))
    ax.set_title("Hidden dimensions (RNN)")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")

    ax.plot(lin, data["rnn_hidden_dim_25"]["train_accs"], color=CB[0],
            label="[25] train")
    ax.plot(lin, data["rnn_hidden_dim_25"]["val_accs"], color=CB[0], linestyle="--",
            label="[25] val")
    ax.plot(lin, data["rnn_hidden_dim_50"]["train_accs"], color=CB[1],
            label="[50] train")
    ax.plot(lin, data["rnn_hidden_dim_50"]["val_accs"], color=CB[1], linestyle="--",
            label="[50] val")
    ax.plot(lin, data["rnn_hidden_dim_100"]["train_accs"], color=CB[2],
            label="[100] train")
    ax.plot(lin, data["rnn_hidden_dim_100"]["val_accs"], color=CB[2], linestyle="--",
            label="[100] val")
    ax.plot(lin, data["rnn_hidden_dim_250"]["train_accs"], color=CB[3],
            label="[250] train")
    ax.plot(lin, data["rnn_hidden_dim_250"]["val_accs"], color=CB[3], linestyle="--",
            label="[250] val")
    ax.legend(loc="upper left")
    plt.savefig("../plots/rnn_hidden_dims.eps", format="eps", dpi=500)
    plt.close()


def plot_dropout():
    """
    Dropout
    """
    cnn_0_0 = "logging.cnn_dropout_0_0.json"
    cnn_0_2 = "logging.cnn_dropout_0_2.json"
    cnn_0_5 = "logging.cnn_dropout_0_5.json"
    rnn_0_0 = "logging.rnn_dropout_0_0.json"
    rnn_0_2 = "logging.rnn_dropout_0_2.json"
    rnn_0_5 = "logging.rnn_dropout_0_5.json"

    data = {}
    for exp in (cnn_0_0, cnn_0_2, cnn_0_5,
                rnn_0_0, rnn_0_2, rnn_0_5):
        data[exp.split(".")[1]] = json.load(open(exp, "r"))

    # CNN
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(45, 85)
    ax.set_xticks(np.arange(0, epochs + 1, 1))
    ax.set_title("Dropout probability (CNN)")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")

    ax.plot(lin, data["cnn_dropout_0_0"]["train_accs"], color=CB[0],
            label="[0.0] train")
    ax.plot(lin, data["cnn_dropout_0_0"]["val_accs"], color=CB[0], linestyle="--",
            label="[0.0] val")
    ax.plot(lin, data["cnn_dropout_0_2"]["train_accs"], color=CB[1],
            label="[0.2] train")
    ax.plot(lin, data["cnn_dropout_0_2"]["val_accs"], color=CB[1], linestyle="--",
            label="[0.2] val")
    ax.plot(lin, data["cnn_dropout_0_5"]["train_accs"], color=CB[2],
            label="[0.5] train")
    ax.plot(lin, data["cnn_dropout_0_5"]["val_accs"], color=CB[2], linestyle="--",
            label="[0.5] val")
    ax.legend(loc="lower right")
    plt.savefig("../plots/cnn_dropout.eps", format="eps", dpi=500)
    plt.close()

    # RNN
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(45, 85)
    ax.set_xticks(np.arange(0, epochs + 1, 1))
    ax.set_title("Dropout probability (RNN)")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")

    ax.plot(lin, data["rnn_dropout_0_0"]["train_accs"], color=CB[0],
            label="[0.0] train")
    ax.plot(lin, data["rnn_dropout_0_0"]["val_accs"], color=CB[0], linestyle="--",
            label="[0.0] val")
    ax.plot(lin, data["rnn_dropout_0_2"]["train_accs"], color=CB[1],
            label="[0.2] train")
    ax.plot(lin, data["rnn_dropout_0_2"]["val_accs"], color=CB[1], linestyle="--",
            label="[0.2] val")
    ax.plot(lin, data["rnn_dropout_0_5"]["train_accs"], color=CB[2],
            label="[0.5] train")
    ax.plot(lin, data["rnn_dropout_0_5"]["val_accs"], color=CB[2], linestyle="--",
            label="[0.5] val")
    ax.legend(loc="lower right")
    plt.savefig("../plots/rnn_dropout.eps", format="eps", dpi=500)
    plt.close()


def plot_kernel_size():
    """
    Kernel size (CNN)
    """
    cnn_1 = "logging.cnn_kernel_size_1.json"
    cnn_2 = "logging.cnn_kernel_size_2.json"
    cnn_3 = "logging.cnn_kernel_size_3.json"

    data = {}
    for exp in (cnn_1, cnn_2, cnn_3):
        data[exp.split(".")[1]] = json.load(open(exp, "r"))

    # CNN
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(50, 85)
    ax.set_xticks(np.arange(0, epochs + 1, 1))
    ax.set_title("Kernel size (CNN)")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")

    ax.plot(lin, data["cnn_kernel_size_1"]["train_accs"], color=CB[0],
            label="[1] train")
    ax.plot(lin, data["cnn_kernel_size_1"]["val_accs"], color=CB[0], linestyle="--",
            label="[1] val")
    ax.plot(lin, data["cnn_kernel_size_2"]["train_accs"], color=CB[1],
            label="[2] train")
    ax.plot(lin, data["cnn_kernel_size_2"]["val_accs"], color=CB[1], linestyle="--",
            label="[2] val")
    ax.plot(lin, data["cnn_kernel_size_3"]["train_accs"], color=CB[2],
            label="[3] train")
    ax.plot(lin, data["cnn_kernel_size_3"]["val_accs"], color=CB[2], linestyle="--",
            label="[3] val")
    ax.legend(loc="upper left")
    plt.savefig("../plots/cnn_kernel_size.eps", format="eps", dpi=500)
    plt.close()


def plot_best():
    """
    Best model (CNN and RNN)
    """
    cnn = "logging.cnn_best.json"
    rnn = "logging.rnn_best.json"

    data = {}
    for exp in (cnn, rnn):
        data[exp.split(".")[1]] = json.load(open(exp, "r"))

    # Adjust for additional epochs
    epochs = 10
    lin = np.linspace(0, epochs, epochs * 3)

    # Accuracy
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(50, 100)
    ax.set_xticks(np.arange(0, epochs + 1, 1))

    ax.set_title("Accuracy (CNN vs. RNN)")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation accuracy")

    ax.plot(lin, data["cnn_best"]["train_accs"], color=CB[0],
            label="[CNN] train")
    ax.plot(lin, data["cnn_best"]["val_accs"], color=CB[0], linestyle="--",
            label="[CNN] val")
    ax.plot(lin, data["rnn_best"]["train_accs"], color=CB[1],
            label="[RNN] train")
    ax.plot(lin, data["rnn_best"]["val_accs"], color=CB[1], linestyle="--",
            label="[RNN] val")
    ax.legend(loc="upper left")
    plt.savefig("../plots/cnn_rnn_acc.eps", format="eps", dpi=500)
    plt.close()

    # Loss
    fig, ax = plt.subplots(1, 1)
    ax.set_ylim(0.25, 1.0)
    ax.set_xticks(np.arange(0, epochs + 1, 1))

    ax.set_title("Loss (CNN vs. RNN)")
    ax.set_xlabel("# of epochs")
    ax.set_ylabel("train/validation loss")

    ax.plot(lin, data["cnn_best"]["train_loss"], color=CB[0],
            label="[CNN] train")
    ax.plot(lin, data["cnn_best"]["val_loss"], color=CB[0], linestyle="--",
            label="[CNN] val")
    ax.plot(lin, data["rnn_best"]["train_loss"], color=CB[1],
            label="[RNN] train")
    ax.plot(lin, data["rnn_best"]["val_loss"], color=CB[1], linestyle="--",
            label="[RNN] val")
    ax.legend(loc="lower left")
    plt.savefig("../plots/cnn_rnn_loss.eps", format="eps", dpi=500)
    plt.close()


if __name__ == "__main__":
    plot_lr()
    plot_hidden_dim()
    plot_dropout()
    plot_kernel_size()
    plot_best()
