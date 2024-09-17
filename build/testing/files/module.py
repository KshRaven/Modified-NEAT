
from build.nn.base import NeatModule
from build.nn.genome import Genome
from build.util.qol import manage_params
from build.util.optimization import timer

import torch
import torch.nn as nn

if __name__ == '__main__':

    DEVICE = 'cuda'
    DTYPE = torch.float32

    INPUTS = 2
    OUTPUTS = 1

    GENOMES = {key: Genome(key) for key in range(10000)}
    MODULE = NeatModule(INPUTS, OUTPUTS, True, DEVICE, DTYPE, nn.Sigmoid())
    MODULE.setup(list(GENOMES.values()))

    print(MODULE)
    print([param for param in MODULE.parameters()])
    test_input = torch.randn(100, len(GENOMES), INPUTS).to(DEVICE)

    def test(**options):
        debug = manage_params(options, 'debug', False)
        test_output = MODULE(test_input)
        if debug:
            print(test_input)
            print(test_output, test_output.shape)

    print(tuple([f'{s:.2e}' for s in timer(test, 10000)]))
    test(debug=True)

