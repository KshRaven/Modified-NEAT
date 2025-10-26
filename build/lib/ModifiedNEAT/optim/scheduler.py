
from ModifiedNEAT.config import Config
from ModifiedNEAT.config.base import Configuration
from copy import deepcopy
from typing import Union

import numpy as np


class Scheduler(object):
    def __init__(self, config: Config, params: Union[str, list[str]] = None):
        self.config = config
        self._config = deepcopy(config)
        if params is None:
            params = ['weight_mutate_power']
        elif isinstance(params, str):
            params = [params]
        self.params: list[str] = params
        self._step_idx = 0

    def get(self, variable: str, config: Config = None) -> Union[int, float, bool, str, None]:
        if config is None:
            config = self._config
        for obj in vars(config).values():
            if isinstance(obj, Configuration):
                for label, var in vars(obj).items():
                    if label == variable:
                        return var
        raise ValueError(f"Variable '{variable}' does not exist in Config")

    def set(self, variable: str, value: Union[int, float, bool, str, None], config: Config = None):
        if config is None:
            config = self.config
        for obj in vars(config).values():
            if isinstance(obj, Configuration):
                for label in vars(obj).keys():
                    if label == variable:
                        setattr(obj, label, value)
                        return
        raise ValueError(f"Variable '{variable}' does not exist in Config")

    def reset(self):
        self._step_idx = 0
        for param in self.params:
            self.set(param, self.get(param, self._config), self.config)

    def modify(self, param: str):
        raise NotImplementedError(f"modify method is not implemented")

    def step(self):
        self._step_idx += 1
        for param in self.params:
            self.modify(param)


class CosineAnnealing(Scheduler):
    def __init__(self, config: Config, period: int, factor: float, params: Union[str, list[str]] = None, warm=False, log=False):
        assert period >= 1
        super(CosineAnnealing, self).__init__(config, params)

        self.period = period
        self.factor = factor
        self._theta = np.linspace(0, (2 if not warm else 1) * np.pi, period)
        self.log_scaling = log

    def modifier(self, param: str):
        value = self.get(param)
        if self.log_scaling:
            if value == 0:
                # raise ValueError(f"0 value encountered during Cosine Log scheduling")
                self.log_scaling = False
        max = value
        min = value * self.factor
        if min > max:
            min, max = max, min
            flipped = True
        else:
            flipped = False
        if self.log_scaling:
            max, min = np.log10(max), np.log10(min)
        a = (max - min) / 2
        b = (max + min) / 2
        theta = self._theta[self._step_idx % self.period]
        new_value = a * np.cos(theta+(np.pi if flipped else 0)) + b
        if self.log_scaling:
            new_value = 10 ** new_value
        self.set(param, new_value)

    def modify(self, param: str):
        self.modifier(param)
        if self._step_idx == self.period:
            self.reset()


class RandomAnnealing(Scheduler):
    def __init__(self, config: Config, min: float, max: float, step=1, params: Union[str, list[str]] = None, log=False):
        assert max > min
        super(RandomAnnealing, self).__init__(config, params)
        for param in self.params:
            assert min <= self.get(param, config) <= max

        self.log_scaling = log
        self.min = min
        self.max = max
        self._step = step

    def modify(self, param: str):
        if self._step_idx % self._step == 0:
            min, max = self.min, self.max
            if self.log_scaling:
                min, max = np.log10(min), np.log10(max)
            diff = max - min
            scale = np.random.rand() * diff
            new_value = min + scale
            if self.log_scaling:
                new_value = 10 ** new_value
            self.set(param, new_value)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    import torch.optim as optim

    matplotlib.use('tkagg')

    CONFIG = Config('scheduler_test')
    CONFIG.genome.weight_mutate_power = 1.0
    CONFIG.reproduction.elitism = 40
    CONFIG.save()
    CONFIG.load(2)

    schedulers = [
        CosineAnnealing(CONFIG, 10, 0.001, warm=True, log=True),
        CosineAnnealing(CONFIG, 20, 1.5, 'elitism', log=True)
    ]

    plot0 = []
    plot1 = []
    for _ in range(52):
        plot0.append(CONFIG.genome.weight_mutate_power)
        plot1.append(CONFIG.reproduction.elitism)

        for s in schedulers:
            s.step()

    _, axes = plt.subplots(2, 1, figsize=(10.8, 7.2))
    axes = axes.flatten()
    axes[0].plot(plot0, label='mutate_power')
    axes[1].plot(plot1, label='elitism')

    plt.legend()
    plt.show()
