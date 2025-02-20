# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import tagger.utils as utils
import tagger.utils.summary as summary


class LearningRateSchedule(object):

    def __call__(self, step):
        raise NotImplementedError("Not implemented.")

    def get_config(self):
        raise NotImplementedError("Not implemented.")

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LinearWarmupRsqrtDecay(LearningRateSchedule):

    def __init__(self, learning_rate, warmup_steps, initial_learning_rate=0.0,
                 summary=True):
        super(LinearWarmupRsqrtDecay, self).__init__()

        if not initial_learning_rate:
            initial_learning_rate = learning_rate / warmup_steps

        self._initial_learning_rate = initial_learning_rate
        self._maximum_learning_rate = learning_rate
        self._warmup_steps = warmup_steps
        self._summary = summary

    def __call__(self, step):
        if step <= self._warmup_steps:
            lr_step = self._maximum_learning_rate - self._initial_learning_rate
            lr_step /= self._warmup_steps
            lr = self._initial_learning_rate + lr_step * step
        else:
            step = step / self._warmup_steps
            lr = self._maximum_learning_rate * (step ** -0.5)

        if self._summary:
            summary.scalar("learning_rate", lr, utils.get_global_step())

        return lr

    def get_config(self):
        return {
            "learning_rate": self._maximum_learning_rate,
            "initial_learning_rate": self._initial_learning_rate,
            "warmup_steps": self._warmup_steps
        }


class PiecewiseConstantDecay(LearningRateSchedule):

    def __init__(self, boundaries, values, summary=True, name=None):
        super(PiecewiseConstantDecay, self).__init__()

        if len(boundaries) != len(values) - 1:
            raise ValueError("The length of boundaries should be 1"
                             " less than the length of values")

        self._boundaries = boundaries
        self._values = values
        self._summary = summary

    def __call__(self, step):
        boundaries = self._boundaries
        values = self._values
        learning_rate = values[0]

        if step <= boundaries[0]:
            learning_rate = values[0]
        elif step > boundaries[-1]:
            learning_rate = values[-1]
        else:
            for low, high, v in zip(boundaries[:-1], boundaries[1:],
                                    values[1:-1]):

                if step > low and step <= high:
                    learning_rate = v
                    break

        if self._summary:
            summary.scalar("learning_rate", learning_rate,
                           utils.get_global_step())

        return learning_rate

    def get_config(self):
        return {
            "boundaries": self._boundaries,
            "values": self._values,
        }


class LinearExponentialDecay(LearningRateSchedule):

    def __init__(self, learning_rate, warmup_steps, start_decay_step,
                 end_decay_step, n, summary=True):
        super(LinearExponentialDecay, self).__init__()

        self._learning_rate = learning_rate
        self._warmup_steps = warmup_steps
        self._start_decay_step = start_decay_step
        self._end_decay_step = end_decay_step
        self._n = n
        self._summary = summary

    def __call__(self, step):
        # See reference: The Best of Both Worlds: Combining Recent Advances
        # in Neural Machine Translation
        n = self._n
        p = self._warmup_steps / n
        s = n * self._start_decay_step
        e = n * self._end_decay_step

        learning_rate = self._learning_rate

        learning_rate *= min(
            1.0 + (n - 1) * step / float(n * p),
            n,
            n * ((2 * n) ** (float(s - n * step) / float(e - s))))

        if self._summary:
            summary.scalar("learning_rate", learning_rate,
                           utils.get_global_step())

        return learning_rate

    def get_config(self):
        return {
            "learning_rate": self._learning_rate,
            "warmup_steps": self._warmup_steps,
            "start_decay_step": self._start_decay_step,
            "end_decay_step": self._end_decay_step,
        }


class LearningRateSchedule(object):

    def __call__(self, step):
        raise NotImplementedError("Not implemented.")

    def get_config(self):
        raise NotImplementedError("Not implemented.")

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class LinearWarmupRsqrtDecay(LearningRateSchedule):

    def __init__(self, learning_rate, warmup_steps, initial_learning_rate=0.0,
                 summary=True):
        super(LinearWarmupRsqrtDecay, self).__init__()

        if not initial_learning_rate:
            initial_learning_rate = learning_rate / warmup_steps

        self._initial_learning_rate = initial_learning_rate
        self._maximum_learning_rate = learning_rate
        self._warmup_steps = warmup_steps
        self._summary = summary

    def __call__(self, step):
        if step <= self._warmup_steps:
            lr_step = self._maximum_learning_rate - self._initial_learning_rate
            lr_step /= self._warmup_steps
            lr = self._initial_learning_rate + lr_step * step
        else:
            step = step / self._warmup_steps
            lr = self._maximum_learning_rate * (step ** -0.5)

        if self._summary:
            summary.scalar("learning_rate", lr, utils.get_global_step())

        return lr

    def get_config(self):
        return {
            "learning_rate": self._maximum_learning_rate,
            "initial_learning_rate": self._initial_learning_rate,
            "warmup_steps": self._warmup_steps
        }


class PiecewiseConstantDecay(LearningRateSchedule):

    def __init__(self, boundaries, values, summary=True, name=None):
        super(PiecewiseConstantDecay, self).__init__()

        if len(boundaries) != len(values) - 1:
            raise ValueError("The length of boundaries should be 1"
                             " less than the length of values")

        self._boundaries = boundaries
        self._values = values
        self._summary = summary

    def __call__(self, step):
        boundaries = self._boundaries
        values = self._values
        learning_rate = values[0]

        if step <= boundaries[0]:
            learning_rate = values[0]
        elif step > boundaries[-1]:
            learning_rate = values[-1]
        else:
            for low, high, v in zip(boundaries[:-1], boundaries[1:],
                                    values[1:-1]):

                if step > low and step <= high:
                    learning_rate = v
                    break

        if self._summary:
            summary.scalar("learning_rate", learning_rate,
                           utils.get_global_step())

        return learning_rate

    def get_config(self):
        return {
            "boundaries": self._boundaries,
            "values": self._values,
        }


class LinearExponentialDecay(LearningRateSchedule):

    def __init__(self, learning_rate, warmup_steps, start_decay_step,
                 end_decay_step, n, summary=True):
        super(LinearExponentialDecay, self).__init__()

        self._learning_rate = learning_rate
        self._warmup_steps = warmup_steps
        self._start_decay_step = start_decay_step
        self._end_decay_step = end_decay_step
        self._n = n
        self._summary = summary

    def __call__(self, step):
        # See reference: The Best of Both Worlds: Combining Recent Advances
        # in Neural Machine Translation
        n = self._n
        p = self._warmup_steps / n
        s = n * self._start_decay_step
        e = n * self._end_decay_step

        learning_rate = self._learning_rate

        learning_rate *= min(
            1.0 + (n - 1) * step / float(n * p),
            n,
            n * ((2 * n) ** (float(s - n * step) / float(e - s))))

        if self._summary:
            summary.scalar("learning_rate", learning_rate,
                           utils.get_global_step())

        return learning_rate

    def get_config(self):
        return {
            "learning_rate": self._learning_rate,
            "warmup_steps": self._warmup_steps,
            "start_decay_step": self._start_decay_step,
            "end_decay_step": self._end_decay_step,
        }


if __name__ == '__main__':
    opts = [LinearWarmupRsqrtDecay(0.5, 1000),
            LinearWarmupRsqrtDecay(0.5, 3000),
            LinearWarmupRsqrtDecay(1, 1000)]
    plt.plot(np.arange(1, 5000), [[opt(i) for opt in opts] for i in range(1, 5000)])
    plt.legend(["0.5:500", "0.5:1000", "1:500"])
    plt.show()