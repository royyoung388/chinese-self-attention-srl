# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import tagger.utils as utils

from tagger.modules.module import Module
from tagger.modules.affine import Affine


class MultiHeadAttention(Module):

    def __init__(self, hidden_size, num_heads, dropout=0.0,
                 name="multihead_attention"):
        super(MultiHeadAttention, self).__init__(name=name)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        with utils.scope(name):
            self.qkv_transform = Affine(hidden_size, 3 * hidden_size,
                                        name="qkv_transform")
            self.o_transform = Affine(hidden_size, hidden_size,
                                      name="o_transform")

        self.reset_parameters()

    def forward(self, query, bias):
        qkv = self.qkv_transform(query)
        q, k, v = torch.split(qkv, self.hidden_size, dim=-1)

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        vh = self.split_heads(v, self.num_heads)

        # scale query
        qh = qh * (self.hidden_size // self.num_heads) ** -0.5

        # dot-product attention
        kh = torch.transpose(kh, -2, -1)
        logits = torch.matmul(qh, kh)

        if bias is not None:
            logits = logits + bias

        weights = torch.nn.functional.dropout(torch.softmax(logits, dim=-1),
                                              p=self.dropout,
                                              training=self.training)

        x = torch.matmul(weights, vh)

        # combine heads
        output = self.o_transform(self.combine_heads(x))

        return output

    def reset_parameters(self, initializer="orthogonal"):
        if initializer == "orthogonal":
            self.qkv_transform.orthogonal_initialize()
            self.o_transform.orthogonal_initialize()
        else:
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.qkv_transform.weight)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.qkv_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)

    @staticmethod
    def split_heads(x, heads):
        batch = x.shape[0]
        length = x.shape[1]
        channels = x.shape[2]

        y = torch.reshape(x, [batch, length, heads, channels // heads])
        return torch.transpose(y, 2, 1)

    @staticmethod
    def combine_heads(x):
        batch = x.shape[0]
        heads = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.transpose(x, 2, 1)

        return torch.reshape(y, [batch, length, heads * channels])
