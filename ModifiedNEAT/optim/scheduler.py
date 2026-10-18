from ModifiedNEAT.config import Config
from ModifiedNEAT.config.base import Configuration
from copy import deepcopy
from typing import Union

import numpy as np
import warnings


class Scheduler(object):
    def __init__(self, config: Config, params: Union[str, list[str]], defaults: dict = None, **options):
        """
        :param config: Config whose values are scheduled.
        :param params: Variable name(s) to schedule (a NEAT config variable or an RL algorithm attribute).
        :param defaults: Initial values for variables that don't exist in any config section yet. Such variables
            are created in ``config.rl`` (and are saved/loaded with the config) instead of raising an error.
            If omitted, the initial value is taken from the algorithm the scheduler gets bound to (see ``bind``).
        :param options: Extra scheduler settings shared by every scheduler; unknown names raise a ``TypeError``.
        :keyword phase: (int, default 0) Number of ``step()`` calls to hold the initial value before the schedule
            actually starts. With ``phase=0`` the first ``step()`` already applies the schedule; with ``phase=10``
            the first 10 calls change nothing and the schedule begins on the 11th. ``reset()`` restarts the phase.
        """
        self.phase: int = int(options.pop('phase', 0) or 0)
        if self.phase < 0: raise ValueError(f"phase must be >= 0, got {self.phase}")
        if options: raise TypeError(f"{self.__class__.__name__} got unexpected options: {sorted(options)}")
        self.config = config
        self._config = deepcopy(config)
        self._defaults = dict(defaults) if defaults else {}
        if params is None: raise ValueError(f"Must set a parameter from the NEAT configuration or RL algo attribute")
        elif isinstance(params, str): params = [params]
        self.params: list[str] = params
        self._step_idx = 0
        self.target = None
        self._pending: set[str] = set()   # RL variables created without a known initial value yet
        for param in self.params:
            if self._find(self.config, param) is None:
                self._register(param)

    @property
    def _t(self) -> int:
        """Steps taken since the phase ended (0 while still in the phase); this is what schedules should index by."""
        return max(0, self._step_idx - self.phase)

    @staticmethod
    def _find(config: Config, variable: str):
        """Return the Configuration section holding ``variable``, or None."""
        for obj in vars(config).values():
            if isinstance(obj, Configuration) and variable in vars(obj):
                return obj
        return None

    def _register(self, variable: str, default=None):
        """Create ``variable`` in the RL section of both the live and the reference config."""
        if default is None:
            default = self._defaults.get(variable)
        if default is None:
            # No starting value yet: use a placeholder until bind() reads it from the algorithm.
            default = 0.0
            self._pending.add(variable)
        for cfg in (self.config, self._config):
            cfg.rl.add(variable, default)

    def bind(self, target):
        """
        Attach the algorithm whose attributes follow this schedule. Called by the algorithm when it receives
        its schedulers. Variables created without a default take the algorithm's current value as their
        starting point; then the config values are pushed onto the algorithm, and again after every
        ``step`` / ``reset``.
        """
        self.target = target
        for param in list(self._pending):
            if hasattr(target, param):
                value = getattr(target, param)
                for cfg in (self.config, self._config):
                    cfg.rl.add(param, value, overwrite=True)
                self._pending.discard(param)
        self._sync()

    def _sync(self):
        if self.target is not None:
            self.config.rl.apply(self.target, names=self.params)

    def get(self, variable: str, config: Config = None) -> Union[int, float, bool, str, None]:
        if config is None:
            config = self._config
        obj = self._find(config, variable)
        if obj is None:
            self._register(variable)
            obj = self._find(config, variable)
        return getattr(obj, variable)

    def set(self, variable: str, value: Union[int, float, bool, str, None], config: Config = None):
        if config is None:
            config = self.config
        obj = self._find(config, variable)
        if obj is None:
            self._register(variable, value)
            obj = self._find(config, variable)
        setattr(obj, variable, value)

    def _restore(self):
        """Put every scheduled variable back to its initial (reference) value."""
        for param in self.params:
            self.set(param, self.get(param, self._config), self.config)

    def reset(self):
        self._step_idx = 0
        self._restore()
        self._sync()

    def modify(self, param: str):
        raise NotImplementedError(f"modify method is not implemented")

    def step(self):
        if self._pending:
            warnings.warn(f"No initial value for {sorted(self._pending)}: not in Config, no defaults given and no "
                          f"algorithm bound with that attribute. Using 0.0.")
            self._pending.clear()
        self._step_idx += 1
        if self._step_idx > self.phase:   # still in the phase: hold the initial values
            for param in self.params:
                self.modify(param)
        self._sync()


class CosineAnnealing(Scheduler):
    def __init__(self, config: Config, period: int, factor: float, params: Union[str, list[str]] = None,
                 warm=False, log=True, defaults: dict = None, **options):
        assert period >= 1
        super(CosineAnnealing, self).__init__(config, params, defaults, **options)

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
        theta = self._theta[self._t % self.period]
        new_value = a * np.cos(theta+(np.pi if flipped else 0)) + b
        if self.log_scaling:
            new_value = 10 ** new_value
        self.set(param, float(new_value))

    def modify(self, param: str):
        self.modifier(param)
        if self._t == self.period:
            self._step_idx = self.phase
            self._restore()


class RandomAnnealing(Scheduler):
    def __init__(self, config: Config, min: float, max: float, step=1, params: Union[str, list[str]] = None,
                 log=True, defaults: dict = None, **options):
        assert max > min
        super(RandomAnnealing, self).__init__(config, params, defaults, **options)
        for param in self.params:
            if param not in self._pending:
                assert min <= self.get(param, config) <= max

        self.log_scaling = log
        self.min = min
        self.max = max
        self._step = step

    def modify(self, param: str):
        if self._t % self._step == 0:
            min, max = self.min, self.max
            if self.log_scaling:
                min, max = np.log10(min), np.log10(max)
            diff = max - min
            scale = np.random.rand() * diff
            new_value = min + scale
            if self.log_scaling:
                new_value = 10 ** new_value
            self.set(param, float(new_value))


class BinaryAnnealing(Scheduler):
    def __init__(self, config: Config, period: int, value: float, params: Union[str, list[str]] = None,
                 defaults: dict = None, **options):
        """
        period: number of steps to hold each state (can be 1)
        value: value applied during the alternate state
        """
        assert period >= 1, "period must be >= 1"
        super(BinaryAnnealing, self).__init__(config, params, defaults, **options)

        self.period = period
        self.value = value

    def modify(self, param: str):
        state = (self._t // self.period) % 2

        if state == 0:
            new_value = self.get(param, self._config)
        else:
            new_value = self.value

        self.set(param, new_value)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib
    import torch.optim as optim

    matplotlib.use('tkagg')

    CONFIG = Config('scheduler_test')
    CONFIG.genome.weight_mutate_power = 1.0
    CONFIG.reproduction.elitism = 0.20
    CONFIG.reproduction.cross_threshold = 0.10
    CONFIG.rl.add('ent_reg', 1.0)
    CONFIG.save()
    CONFIG.load(2)

    schedulers = [
        CosineAnnealing(CONFIG, 10, 0.001, 'weight_mutate_power', warm=True, log=True),
        CosineAnnealing(CONFIG, 20, 1.5, 'elitism', log=True, phase=10),
        BinaryAnnealing(CONFIG, 15, 0, 'cross_threshold', phase=5),
        CosineAnnealing(CONFIG, 25, 0.1, 'ent_reg', log=True),
        # Variable that exists nowhere yet: created in config.rl with the given default
        CosineAnnealing(CONFIG, 30, 0.5, 'my_new_reg', log=False, defaults={'my_new_reg': 2.0}),
    ]

    plots: tuple[list[int | float | bool], ...] = tuple([[] for _ in range(len(schedulers))])
    for _ in range(52):
        data = [
            CONFIG.genome.weight_mutate_power,
            CONFIG.reproduction.elitism,
            CONFIG.reproduction.cross_threshold,
            CONFIG.rl.ent_reg,
            CONFIG.rl.my_new_reg,
        ]
        for p, d in zip(plots, data): p.append(d)

        for s in schedulers:
            s.step()

    CONFIG.update()  # persists the current values, including 'my_new_reg'

    n_rows = len(schedulers)
    _, axes = plt.subplots(n_rows, 1, figsize=(10.8, 9.0))
    axes = axes.flatten()
    for axis, plot, label in zip(axes, plots, ['mutate_power', 'elitism', 'cross_threshold', 'ent_reg', 'my_new_reg']):
        axis.plot(plot, label=label)
        axis.legend()

    plt.show()
