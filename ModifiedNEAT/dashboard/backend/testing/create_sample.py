"""Sample NEAT model for testing the visualizer."""
import sys, os
import pickle
from pathlib import Path

# Ensure the backend module can be imported
TEST_DIR = str(Path(__file__).parent)
print(f"Testing directory = {TEST_DIR}")
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

import ModifiedNEAT as meat
import ModifiedNEAT.nn as mn
import torch.nn as nn


if __name__ == "__main__":
    meat.util.storage.set_storage_location("./")
    
    inputs = 3
    outputs = 10
    dim_size = 64
    models_total = 3
    models = tuple(
        mn.Sequential(
            mn.Linear(inputs, dim_size, bias=True),
            *sum([
            [
                mn.LayerNorm((dim_size,), bias=False),
                nn.SiLU(),
                mn.Linear(dim_size, dim_size, bias=False)
            ]
            for _ in range(3) 
            ], []),
            mn.LayerNorm((dim_size,), bias=False),
            nn.SiLU(),
            mn.Linear(dim_size, outputs, bias=True)
        )
        for _ in range(models_total)
    )
    for m in models: m.eval()

    # NOTE: Run python file from dashboard directory
    save_dir = f"backend/testing/files"
    
    models[0].save(filename='grouped_sample', directory=save_dir, replace=True, debug=True)
    models[0].load(filename='grouped_sample', directory=save_dir, strict=True, debug=True)
    
    config = meat.Config('test', './')
    population = meat.Population(50, models[0], config=config)
    for m in models[1:]: population.absorb_population(meat.Population(50, m, config=config))
    
    for m in models[0].neat_modules():
        print(list(m.parameters(False)))
    
    population.save('grouped_sample', directory=save_dir, replace=True, debug=True)
    population.load('grouped_sample', directory=save_dir, debug=True)
