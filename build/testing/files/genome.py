
from build.nn.genome import Node, Genome
from build.util.conversion import float_to_str

from numba import njit, typeof, int64, optional
from numba.typed import List, Dict

import numpy as np

indexer = optional(int64(0))


# Example usage:
num = 123.456000000789
result = float_to_str(num, zeros_lim=2, dec_lim=5)
print(result)  # Output could be "123.456" or "123.45678" depending on the limits

if __name__ == "__main__":
    print(Dict((i, i**2) for i in range(10)).copy())
    print(float_to_str(76.54))
    print(indexer.key)

    node = Node(0)
    new_node = node.copy()
    node.bias = np.random.randn()
    print(node, node.bias)
    print(new_node, new_node.bias)

    @njit
    def get():
        return List([i for i in range(10)]).copy()

    print(get(), typeof(get()))

    INPUTS = 7
    OUTPUTS = 3

    genomes = {}
    for key in range(10):
        genome = Genome(key)
        genome.add_network(0, INPUTS, OUTPUTS)
        genome.fitness = np.random.rand()
        genomes[key] = genome

    child = Genome(10)
    child.update_from_crossover(genomes[0], genomes[1])

    print(child.networks[0].input_keys)
    print(child.networks[0].output_keys)
    print(child.networks[0])
    print(child.networks[0].build)
    print([w.shape for w in child.networks[0].weights])
    print([w.shape for w in child.networks[0].biases])

