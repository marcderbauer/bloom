from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
from random import shuffle
import time
import math

# Currently declared twice, find a way around this
all_letters = string.ascii_letters + " .,;'-"

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    with open(filename, encoding='utf-8') as some_file:
        return [unicodeToAscii(line.strip()) for line in some_file]

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def save_file(filename, list):
    """
    Quick wrapper around the write function
    """
    with open(filename, "w") as f:
            for line in list:
                f.write(f"{line}\n")

def split_dset(dset, split, shuffle_dset=True):
    # TODO: Included in data class now, can be removed soon.
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