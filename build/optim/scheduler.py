
from build.config import Config
from build.config.base import Configuration
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
        for param in self.params:
            self.set(param, self.get(param, self._config), self.config)

    def modify(self, param: str):
        raise NotImplementedError(f"modify method is not implemented")

    def step(self):
        for param in self.params:
            self.modify(param)


class CosineAnnealing(Scheduler):
    def __init__(self, config: Config, period: int, factor: float, params: Union[str, list[str]] = None, warm=False, log=False):
        assert period >= 1
        super(CosineAnnealing, self).__init__(config, params)

        self.period = period
        self.factor = factor
        self._theta = np.linspace(0, (2 if not warm else 1) * np.pi, period)
        self._step_idx = 0
        self.log_scaling = log

    def modifier(self, param: str):
        value = self.get(param)
        if self.log_scaling:
            if value == 0:
                raise ValueError(f"0 value encountered during Cosine Log scheduling")
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
        self._step_idx += 1
        self.modifier(param)
        if self._step_idx == self.period:
            self.reset()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('tkagg')

    config = Config('scheduler_test')
    config.genome.weight_mutate_power = 1.0
    config.reproduction.elitism = 40
    config.save()
    config.load(2)

    schedulers = [
        CosineAnnealing(config, 10, 0.001, warm=True, log=True),
        CosineAnnealing(config, 20, 1.5, 'elitism', log=True)
    ]

    plot0 = []
    plot1 = []
    for _ in range(52):
        plot0.append(config.genome.weight_mutate_power)
        plot1.append(config.reproduction.elitism)

        for s in schedulers:
            s.step()

    _, axes = plt.subplots(2, 1, figsize=(10.8, 7.2))
    axes = axes.flatten()
    axes[0].plot(plot0, label='mutate_power')
    axes[1].plot(plot1, label='elitism')

    plt.legend()
    plt.show()
