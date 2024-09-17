
from build.population import Population
from build.models.main import MiniFormer
from build.config import Config
from build.util.storage import STORAGE_DIR

import torch
import numpy as np

if __name__ == "__main__":
    DEVICE  = 'cuda'
    DTYPE   = torch.float32

    INPUTS      = 2
    OUTPUTS     = 4
    EMBED_SIZE  = 32
    SEQ_LEN     = 8

    MODEL = MiniFormer(INPUTS, OUTPUTS, EMBED_SIZE, SEQ_LEN, 1, 4, 1, 0.1, False, DEVICE, DTYPE)
    MODEL.eval()

    GENOMES = 10

    config_name = "population_test"
    CONFIG      = Config(config_name)
    # CONFIG.load()
    POPULATION  = Population(GENOMES, MODEL, CONFIG, init_reporter=True)
    CONFIG.save()

    test_input = torch.randn(SEQ_LEN, GENOMES, INPUTS).to(DEVICE)

    def run(population: Population):
        test_output = MODEL(test_input, debug=False)
        test_output2 = torch.argmax(MODEL.infer(test_input, pos_idx=-1, debug=False), -1)[:, :3]
        print(torch.argmax(test_output, -1)[-3:, :5], test_output.shape)
        print(torch.argmax(test_output[:, -3:], -1)[:, :3])
        print(test_output2, test_output2.shape)

        for genome in population.genomes.values():
            genome.fitness = round(np.random.rand(), 4)

    POPULATION.run(run, 10)

