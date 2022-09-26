from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
from random import shuffle


# This was taken from a PyTorch tutorial iirc
all_letters = string.ascii_letters + " .,;'-"

def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def readLines(filename):
    """
    Read a file and split into lines
    """
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]

def save_file(filename, list):
    """
    Quick wrapper around the write function
    """
    with open(filename, "w") as f:
            for line in list:
                f.write(f"{line}\n")

def split_dset(dset, split, shuffle_dset=True):
    """
    Takes percentage that each split is supposed to be at and splits the dataset accodringly.
    input percentages must add to one
    """
    if shuffle_dset:
        shuffle(dset)
    split = round(split * len(dset))
    tr = dset[:split]
    d = dset[split:]
    return tr, d