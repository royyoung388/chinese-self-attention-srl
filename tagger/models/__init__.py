# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tagger.models.deepatt
import tagger.models.lstmatt


def get_model(name):
    name = name.lower()

    if name == "deepatt":
        return tagger.models.deepatt.DeepAtt
    elif name == 'lstmatt':
        return tagger.models.lstmatt.LSTMAtt
    else:
        raise LookupError("Unknown model %s" % name)
