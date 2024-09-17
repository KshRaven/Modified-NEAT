
from build.models.main import MiniFormer
from build.nn.base import get_modules
from build.nn.genome import Genome
from build.util.qol import manage_params
from build.util.optimization import timer
from build.nn.activations import Tanh

import torch
import torch.nn as nn

if __name__ == '__main__':
    DEVICE = 'cuda'
    DTYPE = torch.float32

    INPUTS = 2
    OUTPUTS = 3
    EMBED_SIZE = 8
    SEQ_LEN = 16

    GENOMES = {key: Genome(key) for key in range(100000)}
    MODULE = MiniFormer(INPUTS, OUTPUTS, EMBED_SIZE, SEQ_LEN, 1, 1, 1, 0.1, False, DEVICE, DTYPE, pri_actv=Tanh())
    print(MODULE)
    modules = get_modules(MODULE)
    print(modules)
    for module in modules:
        module.setup(list(GENOMES.values()), True)

    # print([(param, param.shape) for param in MODULE.parameters()])
    test_input = torch.randn(3, SEQ_LEN, len(GENOMES), INPUTS).to(DEVICE)
    # test_mask = torch.randint(0, 2, (len(GENOMES),), device=DEVICE, dtype=torch.bool)
    # print(f"sum = {torch.sum(test_mask)} shape = {test_mask.shape}")
    test_mask = None

    def test(**options):
        with torch.no_grad():
            debug = manage_params(options, 'verbose', None)
            test_output = MODULE(test_input, genome_mask=test_mask, verbose=False)
            if debug:
                # print(test_input)
                test_output2 = torch.argmax(MODULE.infer(test_input, pos_idx=-1, genome_mask=test_mask, verbose=False), -1)
                print(torch.argmax(test_output, -1), test_output.shape)
                print(torch.argmax(test_output[:, [-1]], -1))
                print(test_output2, test_output2.shape)

    test(verbose=1)
    print(tuple([f'{s:.2e}' for s in timer(test, 10000)]))

