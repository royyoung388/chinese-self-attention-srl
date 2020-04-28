# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import torch


def oldest_checkpoint(path):
    names = glob.glob(os.path.join(path, "*.pt"))

    if not names:
        return None

    oldest_counter = 10000000
    checkpoint_name = names[0]

    for name in names:
        counter = name.rstrip(".pt").split("-")[-1]

        if not counter.isdigit():
            continue
        else:
            counter = int(counter)

        if counter < oldest_counter:
            checkpoint_name = name
            oldest_counter = counter

    return checkpoint_name


def best_checkpoint(path):
    if not os.path.exists(os.path.join(path, "checkpoint")):
        return latest_checkpoint(path)

    with open(os.path.join(path, "checkpoint")) as fd:
        line = fd.readline()
        name = line.strip().split()[-1][1:-1]

        return os.path.join(path, name)


def latest_checkpoint(path):
    names = glob.glob(os.path.join(path, "*.pt"))

    if not names:
        return None

    latest_counter = 0
    checkpoint_name = names[0]

    for name in names:
        counter = name.rstrip(".pt").split("-")[-1]

        if not counter.isdigit():
            continue
        else:
            counter = int(counter)

        if counter > latest_counter:
            checkpoint_name = name
            latest_counter = counter

    return checkpoint_name


def save(state, path, max_to_keep=None):
    checkpoints = glob.glob(os.path.join(path, "*.pt"))

    if max_to_keep and len(checkpoints) >= max_to_keep:
        checkpoint = oldest_checkpoint(path)
        os.remove(checkpoint)

    if not checkpoints:
        counter = 1
    else:
        checkpoint = latest_checkpoint(path)
        counter = int(checkpoint.rstrip(".pt").split("-")[-1]) + 1

    checkpoint = os.path.join(path, "model-%d.pt" % counter)
    print("Saving checkpoint: %s" % checkpoint)
    torch.save(state, checkpoint)
