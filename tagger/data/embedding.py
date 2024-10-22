# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cgitb

import numpy as np


def load_embedding(filename, vocab=None):
    fd = open(filename, "r")
    emb = {}
    fan_out = 0

    for line in fd:
        items = line.strip().split()
        if len(items) <= 1:
            print(items)
        try:
            word = items[0].encode("utf-8")
            value = [float(item) for item in items[1:]]
        except ValueError as e:
            print('load embedding error:', e, line[:10])
            continue

        fan_out = len(value)
        emb[word] = np.array(value, "float32")

    if not vocab:
        return emb

    ivoc = {}

    for item in vocab:
        ivoc[vocab[item]] = item

    new_emb = np.zeros([len(ivoc), fan_out], "float32")

    for i in ivoc:
        word = ivoc[i]
        if word not in emb:
            fan_in = len(ivoc)
            scale = 3.0 / max(1.0, (fan_in + fan_out) / 2.0)
            new_emb[i] = np.random.uniform(-scale, scale, [fan_out])
        else:
            new_emb[i] = emb[word]

    return new_emb


# def load_fasttext(filename, vocab=None, dim=300):
#     corpus_file = datapath('lee_background.cor')  # absolute path to corpus
#     model = FastText(size=dim, window=3, min_count=3)
#     model.build_vocab(corpus_file=corpus_file)  # scan over corpus to build the vocabulary
#     total_words = model.corpus_total_words  # number of words in the corpus
#     model.train(corpus_file=corpus_file, total_words=total_words, epochs=5)
#
#     ivoc = {}
#
#     if vocab:
#         for item in vocab:
#             ivoc[vocab[item]] = item
#
#     new_emb = np.zeros([len(ivoc), dim], "float32")
#
#     for i in ivoc:
#         word = ivoc[i]
#         if word not in model.wv.vocab:
#             fan_in = len(ivoc)
#             scale = 3.0 / max(1.0, (fan_in + dim) / 2.0)
#             new_emb[i] = np.random.uniform(-scale, scale, [dim])
#         else:
#             new_emb[i] = model.wv[word]
#
#     return new_emb


if __name__ == '__main__':
    cgitb.enable(format='text')
    # load_embedding("/home/roy/Tagger/data/glove/embedding.txt")
    # load_fasttext("/home/roy/Tagger/data/glove/embedding.txt")
