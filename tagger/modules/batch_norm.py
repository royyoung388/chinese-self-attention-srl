# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers

import torch
import torch.nn as nn

import tagger.utils as utils
from tagger.modules.module import Module


class BatchNorm(Module):

    def __init__(self, normalized_shape, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 name="layer_norm"):
        super(BatchNorm, self).__init__(name=name)
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        with utils.scope(name):
            if self.affine:
                self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
                self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
                self.add_name(self.weight, "weight")
                self.add_name(self.bias, "bias")
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)

            if self.track_running_stats:
                self.register_buffer('running_mean', torch.zeros(*normalized_shape))
                self.register_buffer('running_var', torch.ones(*normalized_shape))
                self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
            else:
                self.register_parameter('running_mean', None)
                self.register_parameter('running_var', None)
                self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return nn.functional.batch_norm(
            input.transpose(1, 2), self.running_mean, self.running_var, self.weight, self.bias,
            self.training or not self.track_running_stats,
            self.momentum, self.eps).transpose(1,2)

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)


if __name__ == '__main__':
    bn = BatchNorm(20)
    input = torch.randn(2, 10, 20)
    output = bn(input)
    print(input, input.shape)
    print(output, output.shape)
