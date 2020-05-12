# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

import tagger.modules as modules
import tagger.utils as utils
from tagger.data import load_embedding
from tagger.modules import LSTMCell


class LSTMAtt(modules.Module):

    def __init__(self, params, name="deepatt"):
        super(LSTMAtt, self).__init__(name=name)
        self.params = params

        with utils.scope(name):
            self.build_embedding(params)
            self.encoding = modules.PositionalEmbedding()
            self.encoder = LSTMAttEncoder(params)
            self.classifier = modules.Affine(params.hidden_size,
                                             len(params.vocabulary["target"]),
                                             name="softmax")

        self.criterion = modules.SmoothedCrossEntropyLoss(
            params.label_smoothing)
        self.dropout = params.residual_dropout
        self.hidden_size = params.hidden_size
        self.reset_parameters()

    def build_embedding(self, params):
        vocab_size = len(params.vocabulary["source"])

        self.embedding = torch.nn.Parameter(
            torch.empty([vocab_size, params.feature_size]))
        self.weights = torch.nn.Parameter(
            torch.empty([2, params.feature_size]))
        self.bias = torch.nn.Parameter(torch.zeros([params.hidden_size]))
        self.add_name(self.embedding, "embedding")
        self.add_name(self.weights, "weights")
        self.add_name(self.bias, "bias")

    def reset_parameters(self):
        nn.init.normal_(self.embedding, mean=0.0,
                        std=self.params.feature_size ** -0.5)
        nn.init.normal_(self.weights, mean=0.0,
                        std=self.params.feature_size ** -0.5)
        nn.init.normal_(self.classifier.weight, mean=0.0,
                        std=self.params.hidden_size ** -0.5)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features, labels):
        mask = torch.ne(features["inputs"], 0).float().cuda()
        logits = self.encode(features)
        loss = self.criterion(logits, labels)
        mask = mask.to(logits)

        return torch.sum(loss * mask) / torch.sum(mask)

    def encode(self, features):
        seq = features["inputs"]
        pred = features["preds"]
        mask = torch.ne(seq, 0).float().cuda()
        enc_attn_bias = self.masking_bias(mask)

        inputs = torch.nn.functional.embedding(seq, self.embedding)

        if "embedding" in features and not self.training:
            embedding = features["embedding"]
            unk_mask = features["mask"].to(mask)[:, :, None]
            inputs = inputs * unk_mask + (1.0 - unk_mask) * embedding

        preds = torch.nn.functional.embedding(pred, self.weights)
        inputs = torch.cat([inputs, preds], axis=-1)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias

        inputs = nn.functional.dropout(self.encoding(inputs), self.dropout,
                                       self.training)

        enc_attn_bias = enc_attn_bias.to(inputs)
        encoder_output = self.encoder(inputs, enc_attn_bias, mask)
        logits = self.classifier(encoder_output)

        return logits

    def argmax_decode(self, features):
        logits = self.encode(features)
        return torch.argmax(logits, -1)

    def load_embedding(self, path):
        if not path:
            return
        emb = load_embedding(path, self.params.lookup["source"])

        with torch.no_grad():
            self.embedding.copy_(torch.tensor(emb))

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def base_params():
        params = utils.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            feature_size=100,
            predicate_size=100,
            hidden_size=200,
            filter_size=800,
            num_heads=8,
            num_hidden_layers=10,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            lstm_dropout=0.0,
            label_smoothing=0.1,
            clip_grad_norm=0.0
        )

        return params

    @staticmethod
    def default_params(name=None):
        return LSTMAtt.base_params()


class LSTMAttEncoder(modules.Module):

    def __init__(self, params, name="encoder"):
        super(LSTMAttEncoder, self).__init__(name=name)

        with utils.scope(name):
            self.layers = nn.ModuleList([
                LSTMAttEncoderLayer(params, name="layer_%d" % i)
                for i in range(params.num_hidden_layers)])

    def forward(self, x, bias, mask):
        for layer in self.layers:
            x = layer(x, bias, mask)
        return x


class LSTMAttEncoderLayer(modules.Module):

    def __init__(self, params, name="layer"):
        super(LSTMAttEncoderLayer, self).__init__(name=name)

        with utils.scope(name):
            self.lstm = LSTMSubLayer(params)
            self.self_attention = AttentionSubLayer(params)

    def forward(self, x, bias, mask):
        x = self.lstm(x, mask)
        x = self.self_attention(x, bias)
        return x


class AttentionSubLayer(modules.Module):

    def __init__(self, params, name="attention"):
        super(AttentionSubLayer, self).__init__(name=name)

        with utils.scope(name):
            self.attention = modules.MultiHeadAttention(
                params.hidden_size, params.num_heads, params.attention_dropout)
            self.layer_norm = modules.LayerNorm(params.hidden_size)

        self.dropout = params.residual_dropout

    def forward(self, x, bias):
        y = self.attention(x, bias)
        y = nn.functional.dropout(y, self.dropout, self.training)

        return self.layer_norm(x + y)


class LSTMSubLayer(modules.Module):

    def __init__(self, params, dtype=None, name="ffn_layer"):
        super(LSTMSubLayer, self).__init__(name=name)

        with utils.scope(name):
            self.lstm_forward = LSTMCell(params.hidden_size, params.hidden_size // 2,
                                         normalization=True)
            self.lstm_back = LSTMCell(params.hidden_size, params.hidden_size // 2,
                                      normalization=True)
            self.layer_norm = modules.LayerNorm(params.hidden_size)
        self.dropout = params.lstm_dropout

    def forward(self, x, mask):
        """

        :param x: batch * seq * hidden_size
        :param mask: batch * seq. have word = 1
        :return:
        """
        shape = x.shape

        c_for = torch.zeros([shape[0], shape[2] // 2]).cuda()
        h_for = torch.zeros([shape[0], shape[2] // 2]).cuda()
        c_back = torch.zeros([shape[0], shape[2] // 2]).cuda()
        h_back = torch.zeros([shape[0], shape[2] // 2]).cuda()
        y = torch.empty(shape).cuda()

        for i, item in enumerate(x.unbind(1)):
            y_for, (c_for, h_for) = self.lstm_forward(x[:, i], (c_for, h_for), mask[:, i])
            y_back, (c_back, h_back) = self.lstm_back(x[:, i], (c_back, h_back), mask[:, i])
            # batch * hidden
            y[:, i] = torch.cat((y_for, y_back), 1)

        y = nn.functional.dropout(y, self.dropout, self.training)

        return self.layer_norm(x + y)
