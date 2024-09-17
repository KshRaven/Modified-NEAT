
from build.nn import NeatModule
from build.nn.genome import Genome

from torch import Tensor
from numba import njit

import torch
import numpy as np


FRACT_PV  = torch.arange(-1, -52-1, -1, dtype=torch.float64)
EXP_PV    = torch.arange(10, 0-1, -1, dtype=torch.float64)
FRACT_SEL = (64 - 52) + torch.arange(52, dtype=torch.int)
EXP_SEL   = torch.arange(1, 12, dtype=torch.int)


def _float64_frac(binary_tensor: Tensor):
    index_select = FRACT_SEL.to(binary_tensor.device)
    place_values = FRACT_PV.to(binary_tensor.device)
    return 1 + torch.sum(torch.index_select(binary_tensor, -1, index_select) * (2 ** place_values), -1)


def _float64_exp(binary_tensor: Tensor):
    index_select = EXP_SEL.to(binary_tensor.device)
    place_values = EXP_PV.to(binary_tensor.device)
    return 2 ** (torch.sum(torch.index_select(binary_tensor, -1, index_select) * (2 ** place_values), -1) - 1023)


def _to_float64(binary_tensor: Tensor):
    assert binary_tensor.shape[-1] == 64
    torch.cuda.empty_cache()
    with torch.no_grad():
        return ((-1.0) ** torch.select(binary_tensor, dim=-1, index=0)) * \
            _float64_exp(binary_tensor) * _float64_frac(binary_tensor)


def _to_binary(tensor: Tensor):
    sign = (tensor < 0.0).unsqueeze(-1)
    log = torch.log2(torch.abs(tensor)) + 1023
    exponent = torch.floor(log)
    fraction = (2 ** (log - exponent)) - 1

    # Get exponent in binary
    exponent_bin = []
    for _ in range(11):
        exponent_bin.append((exponent % 2 != 0).unsqueeze(-1))
        exponent = torch.floor(exponent / 2)
    exponent_bin = torch.cat(list(reversed(exponent_bin)), -1)

    # Get fraction in binary
    fraction_bin = []
    for _ in range(52):
        value = fraction * 2
        integer = np.floor(value)
        fraction_bin.append(integer.unsqueeze(-1))
        fraction = value - integer
    fraction_bin = torch.cat(fraction_bin, -1)

    return torch.cat([sign, exponent_bin, fraction_bin], -1)


@njit
def _get_random_bool_array(size: int, prob: float, max_noise=0.10):
    assert size > 0
    indices    = np.arange(size)
    np.random.shuffle(indices)
    bool_array = np.full(size, True, bool)
    prob -= np.random.rand() * max_noise
    limit = np.round(size * (1 - prob))
    bool_array[indices[:limit]] = False
    return bool_array


def _get_random_choice(tensor: Tensor, parent1: Genome, parent2: Genome, max_noise=0.10) -> Tensor:
    minimum = min(0, parent1.fitness, parent2.fitness)
    p1 = parent1.fitness - minimum + 1e-12
    p2 = parent2.fitness - minimum + 1e-12
    p1_prob = p1 / (p1 + p2)
    probs = _get_random_bool_array(tensor.numel(), p1_prob, max_noise)
    probs = torch.tensor(probs, device=tensor.device, dtype=torch.bool).view(tensor.shape)
    return probs


class ReproductionMethods:
    @staticmethod
    def value_crossover(child: Genome, parent1: Genome, parent2: Genome,
                        build: list[NeatModule], source: list[NeatModule], epsilon=0.0):
        with torch.no_grad():
            if parent1.fitness > parent2.fitness:
                parent1, parent2 = parent1, parent2
            else:
                parent1, parent2 = parent2, parent1

            for module_mod, module in zip(build, source):
                # Inherit connection genes (Connections are the weights)
                conn1 = module.weights[parent1.index]
                conn2 = module.weights[parent2.index]
                # Copy Excess or disjoint connection genes from fittest parent
                disjoint = conn2 <= epsilon
                module_mod.weights[child.index][disjoint] = conn1[disjoint]
                # Combine homologous connections genes from both parents
                homologous = torch.logical_not(disjoint)
                gene_crossover = _get_random_choice(conn1, parent1, parent2)
                g1c = torch.logical_and(homologous, gene_crossover)
                g2c = torch.logical_and(homologous, ~gene_crossover)
                # sys.exit(1)
                module_mod.weights[child.index][g1c] = conn1[g1c]
                module_mod.weights[child.index][g2c] = conn2[g2c]

                # Inherit node genes (Connections are the outputs, which mainly exists physically as the biases)
                if module.biases is not None:
                    bias1 = module.biases[parent1.index]
                    bias2 = module.biases[parent2.index]
                    # Copy Excess or disjoint connection genes from fittest parent
                    disjoint = bias2 <= epsilon
                    module_mod.biases[child.index][disjoint] = bias1[disjoint]
                    # Combine homologous connections genes from both parents
                    homologous = torch.logical_not(disjoint)
                    gene_crossover = _get_random_choice(bias1, parent1, parent2)
                    g1b = torch.logical_and(homologous, gene_crossover)
                    g2b = torch.logical_and(homologous, ~gene_crossover)
                    module_mod.biases[child.index][g1b] = bias1[g1b]
                    module_mod.biases[child.index][g2b] = bias2[g2b]

    @staticmethod
    def binary_crossover(child: Genome, parent1: Genome, parent2: Genome,
                         build: list[NeatModule], source: list[NeatModule]):
        with torch.no_grad():
            if parent1.fitness > parent2.fitness:
                parent1, parent2 = parent1, parent2
            else:
                parent1, parent2 = parent2, parent1

            for module_mod, module in zip(build, source):
                # Inherit connection genes (Connections are the weights)
                conn1  = _to_binary(module.weights[parent1.index])
                conn2  = _to_binary(module.weights[parent2.index])
                weight = _to_binary(module_mod.weights[child.index])
                # Combine connections genes from both parents
                gene_crossover = _get_random_choice(conn1, parent1, parent2)
                weight[gene_crossover]  = conn1[gene_crossover]
                weight[~gene_crossover] = conn2[~gene_crossover]
                module_mod.weights[child.index] = _to_float64(weight)

                # Inherit node genes (Connections are the outputs, which mainly exists physically as the biases)
                if module.biases is not None:
                    bias1 = _to_binary(module.biases[parent1.index])
                    bias2 = _to_binary(module.biases[parent2.index])
                    bias  = _to_binary(module_mod.biases[child.index])
                    # Combine node genes from both parents
                    gene_crossover = _get_random_choice(bias1, parent1, parent2)
                    bias[gene_crossover]  = bias1[gene_crossover]
                    bias[~gene_crossover] = bias2[~gene_crossover]
                    module_mod.biases[child.index] = _to_float64(bias)


if __name__ == '__main__':
    from copy import deepcopy

    print(f"fraction index select =>\n{FRACT_SEL}")
    print(f"fraction place values =>\n{FRACT_PV}")
    print(f"exponent index select =>\n{EXP_SEL}")
    print(f"exponent place values =>\n{EXP_PV}")

    test_source  = [NeatModule(10, 3, 1, True)]
    test_module  = deepcopy(test_source)
    test_parent1 = Genome(1, 0)
    test_parent2 = Genome(2, 1)
    test_child   = Genome(3, 2)

    test_source[0].initialize()
    test_parent1.fitness = 100
    test_parent2.fitness = 50

    print(f"Parent1 =>\n{test_source[0].weights[test_parent1.index]}\n{test_source[0].biases[test_parent1.index]}")
    print(f"Parent2 =>\n{test_source[0].weights[test_parent2.index]}\n{test_source[0].biases[test_parent2.index]}")
    print(f"Init Child =>\n{test_module[0].weights[test_child.index]}\n{test_module[0].biases[test_child.index]}")

    ReproductionMethods.binary_crossover(test_child, test_parent1, test_parent2, test_module, test_source)

    print(f"Child =>\n{test_module[0].weights[test_child.index]}\n{test_module[0].biases[test_child.index]}")
    print(f"Parent3 =>\n{test_source[0].weights[test_child.index]}\n{test_source[0].biases[test_child.index]}")
