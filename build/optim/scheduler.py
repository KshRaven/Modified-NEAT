
from build.config import Config
from typing import Any
from copy import deepcopy

import numpy as np


class Scheduler(object):
    def __init__(self, config: Config):
        self.config = config
        self.params: tuple[Any] = tuple([])

    def set(self, *params):
        self.params = params

    def modify(self, *params):
        raise NotImplementedError(f"modify method is not implemented")

    def step(self):
        self.modify(*self.params)


class CosineAnnealing(Scheduler):
    def __init__(self, config: Config, steps: int, period: int, reduction: float = None, warm=False):
        super(CosineAnnealing, self).__init__(config)
        assert period >= 1
        if reduction is None:
            reduction = 0.1 if period < 100 else 0.01

        self.config_ = deepcopy(self.config)
        self._steps = steps
        self._period = period
        self._reduction = reduction
        self._theta = np.linspace(0, (2 if not warm else 1) * np.pi, period)
        self._step_idx = 0

    def modifier(self, value: float):
        a = (value - value * self._reduction) / 2
        b = (value + value * self._reduction) / 2
        theta = self._theta[self._step_idx % self._period]
        return a * np.cos(theta) + b

    def modify(self, *params):
        if self._step_idx == self._steps:
            self._step_idx = 0
            raise ValueError(f"End of scheduler steps")
        self.config.genome.weight_mutate_power = self.modifier(self.config_.genome.weight_mutate_power)
        self.config.genome.bias_mutate_power   = self.modifier(self.config_.genome.bias_mutate_power)
        self._step_idx += 1
        if self._step_idx == self._steps:
            self.config.genome.weight_mutate_power = self.config_.genome.weight_mutate_power
            self.config.genome.bias_mutate_power   = self.config_.genome.bias_mutate_power
