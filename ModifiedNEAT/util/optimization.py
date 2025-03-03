
import time as clock
import numpy as np


def timer(func, epochs: int, steps=1, **kwargs):
    runs = []
    for epochs in range(epochs):
        steps_ = []
        for step in range(steps):
            ts = clock.perf_counter()
            func(**kwargs)
            steps_.append(clock.perf_counter()-ts)
        runs.append(np.mean(steps_))
    runs = runs[1:]

    return np.mean(runs), np.std(runs), np.max(runs), np.min(runs)
