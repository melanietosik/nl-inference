import os
import pickle
import random

import pandas as pd
from collections import Counter

import torch
from torch.utils.data import Dataset

import constants as const


class SNLIDataSet(Dataset):
    """
    Class to represent training and validation dataset in PyTorch format

    NB: inherits torch.utils.data.Dataset
    """

    def __init__(self, data_fp, tok2id):
        """
        @param data_fp: path to data file (training or validation)
        @param tok2id: token-to-index mapping
        """
        self.df = pd.read_csv(data_fp, sep="\t")
        self.tok2id = tok2id
        self.label_map = {
            "contradiction": 0,
            "entailment": 1,
            "neutral": 2,
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        """
        Convert sentences to integer representations

        @param ix: index of sample in data file
        @return prem_ix: integer mapping of "sentence1"
        @return hypo_ix: integer mapping of "sentence2"
        @return label_ix: index of label [0, 1, 2]
        """
        prem_toks = self.df.iloc[ix]["sentence1"].lower().split()
        hypo_toks = self.df.iloc[ix]["sentence2"].lower().split()
        label = self.df.iloc[ix]["label"].lower()

        prem_ix = [self.tok2id.get(tok, const.UNK_IDX) for tok in prem_toks]
        hypo_ix = [self.tok2id.get(tok, const.UNK_IDX) for tok in hypo_toks]
        label_ix = self.label_map[label]

        return (
            prem_ix[:const.MAX_SENT_LEN],
            hypo_ix[:const.MAX_SENT_LEN],
            label_ix)


def collate_fn(batch):
    """
    Custom DataLoader() function to dynamically pad the batch
      so that all samples have the same length
    """
    # Unpack batch tensor
    prems, hypos, labels = zip(*batch)

    # Get max. sentence lengths
    prem_len = max(len(p) for p in prems)
    hypo_len = max(len(h) for h in hypos)

    # Build output tensors, fill completely with pads
    prems_mat = torch.LongTensor(
        len(prems), prem_len).fill_(const.PAD_IDX)
    hypos_mat = torch.LongTensor(
        len(hypos), hypo_len).fill_(const.PAD_IDX)

    # Put token indices into the tail end of the matrix
    for i, (prem, hypo) in enumerate(zip(prems, hypos)):
        prems_mat[i, -len(prem):] = torch.LongTensor(prem)
        hypos_mat[i, -len(hypo):] = torch.LongTensor(hypo)

    labels = torch.LongTensor(labels)
    return prems_mat, hypos_mat, labels


def build_vocab(train_toks):
    """
    Build vocabulary

    @param train_toks: list of all tokens in training set
    @return id2tok: list of tokens; id2tok[i] returns token at i
    @return tok2id: dictionary; keys are tokens, values are indices
    """
    tok_cnt = Counter(train_toks)  # 21023 unique tokens
    vocab, cnt = zip(*tok_cnt.most_common(const.MAX_VOCAB_SIZE - 2))

    id2tok = list(vocab)
    tok2id = dict(zip(vocab, range(2, 2 + len(vocab))))

    id2tok = ["<pad>", "<unk>"] + id2tok
    tok2id["<pad>"] = const.PAD_IDX
    tok2id["<unk>"] = const.UNK_IDX

    return tok2id, id2tok


def build_or_load_vocab(train_fp, overwrite=False):
    """
    Build vocabulary or load from file

    @param args: command line arguments
    @param overwrite: overwrite existing token-index mappings
    """
    ID2TOK = const.ID2TOK
    TOK2ID = const.TOK2ID

    if ((not overwrite) and os.path.exists(ID2TOK) and os.path.exists(TOK2ID)):
        id2tok = pickle.load(open(ID2TOK, "rb"))
        tok2id = pickle.load(open(TOK2ID, "rb"))
        return tok2id, id2tok

    df = pd.read_csv(train_fp, sep="\t")

    # Extract text and lowercase
    sentence1 = df["sentence1"].values
    sentence1_text = " ".join(x.lower() for x in sentence1)
    sentence2 = df["sentence2"].values
    sentence2_text = " ".join(x.lower() for x in sentence2)
    text = sentence1_text + " " + sentence2_text

    # Build vocabulary
    tok2id, id2tok = build_vocab(text.split())

    # Save to files
    pickle.dump(id2tok, open(ID2TOK, "wb"))
    pickle.dump(tok2id, open(TOK2ID, "wb"))

    assert(len(tok2id) == len(id2tok))

    # Check random token/index
    rand_tok_id = random.randint(0, len(id2tok) - 1)
    rand_tok = id2tok[rand_tok_id]
    assert(rand_tok_id == tok2id[rand_tok])
    assert(rand_tok == id2tok[rand_tok_id])

    return tok2id, id2tok
