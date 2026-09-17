
import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)

from ModifiedNEAT.multproc import ModelWrapper, Processor, verbose
from ModifiedNEAT import nn as mn
from ModifiedNEAT import Population, Config

import torch
import torch.nn as nn
import timeit
import numpy as np

from numba.core.errors import NumbaPerformanceWarning

warnings.filterwarnings(action="ignore", category=NumbaPerformanceWarning)


DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64


def test():
    genomes         = 100
    model_num       = 4
    inputs          = 3
    outputs         = 1
    hidden_layers   = 3
    dim_size        = 512
    bias            = True
    models = [
        mn.Sequential(
            mn.Linear(inputs, dim_size, bias, DEVICE, DTYPE),
            nn.Tanh(),
            *sum([
                [
                    mn.Linear(dim_size, dim_size, bias, DEVICE, DTYPE),
                    nn.Tanh(),
                ]
                for _ in range(hidden_layers)
            ], []),
            mn.Linear(dim_size, outputs, bias, DEVICE, DTYPE),
        )
        for _ in range(model_num)
    ]
    print(models[0])
    config = Config('parallelism')
    populations = [Population(genomes, m, config, verbose=False) for m in models]

    print("starting")
    wrappers = [ModelWrapper(m, (genomes, inputs,), (outputs,), DEVICE, DTYPE) for m in models]
    processor = Processor(models, device=DEVICE)
    print(processor.method, processor.device)
    # processor.set_start_method('spawn', True)

    def test0():
        with verbose(1):
            test_tensors = [torch.randn((genomes, 1, inputs,), device=DEVICE, dtype=DTYPE) for m in processor.models]
            test_outputs = [t.shape for t in processor.run(test_tensors)]
            print(test_outputs)
    test0()

    def test1():
        test_tensors = [torch.randn((genomes, 1, inputs,), device=DEVICE, dtype=DTYPE) for m in processor.models]
        test_outputs = [m(t) for m, t in zip(models, test_tensors)]
        test_outputs = [t.shape for t in test_outputs]
    test1()

    def test2():
        # with verbose(0):
        test_tensors = [torch.randn((genomes, 1, inputs,), device=DEVICE, dtype=DTYPE) for m in processor.models]
        test_outputs = [t.shape for t in processor.run(test_tensors)]

    def test3():
        test_tensors = [torch.randn((genomes, 1, inputs,), device=DEVICE, dtype=DTYPE) for m in processor.models]
        test_outputs = [t.shape for t in processor(test_tensors)]

    print(f"time_refr = {np.mean(timeit.repeat(test1, repeat=10, number=3))}")
    print(f"time_main = {np.mean(timeit.repeat(test2, repeat=10, number=3))}")
    print(f"time_norm = {np.mean(timeit.repeat(test3, repeat=10, number=3))}")

    for trial_idx in range(3):
        print(f"\nTrial {trial_idx}")
        test0()


if __name__ == '__main__':
    test()
