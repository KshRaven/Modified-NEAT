# Configuration guide

ModifiedNEAT uses a JSON-backed configuration system to persist experiment settings for evolution runs. The main entry point is the Config wrapper exposed by the package.

## Main classes

The configuration stack is implemented in the config package:

- Configuration: base helper for loading, updating, and saving section data
- GeneralConfig: global run settings
- GenomeConfig: mutation, initialization, and weight settings
- SpeciesConfig: species compatibility rules
- StagnationConfig: rules for species stagnation
- ReproductionConfig: reproduction and selection behavior
- Config: top-level wrapper that groups all sections together

## Creating and saving a config

```python
import ModifiedNEAT as neat

config = neat.Config(file_name="demo", directory="demo")
config.general.pop_size = 120
config.genome.weight_mutate_rate = 0.5
config.genome.weight_mutate_power = 0.2
config.save(debug=True)
```

This stores the configuration as a JSON file under the storage directory for the selected experiment folder.

## Section overview

### GeneralConfig

Controls the overall evolution loop:

- fitness criterion and threshold
- population size
- reset-on-extinction behavior
- random seed

### GenomeConfig

Controls how genomes are initialized and mutated:

- weight initialization mean and standard deviation
- weight mutation power and rate
- add/delete mutation probabilities
- structural mutation behavior
- compatibility scaling factors

### SpeciesConfig

Controls speciation:

- compatibility threshold for assigning genomes to species

### StagnationConfig

Controls how species are handled when progress stalls:

- species fitness function
- maximum stagnation period
- elitism within species

### ReproductionConfig

Controls reproduction and selection:

- elitism ratio
- clone threshold
- survival threshold
- crossover behavior and multipliers
- minimum species size
- purge behavior

## Practical notes

- Use descriptive config names for each experiment.
- Call save() after changing values if you want the settings persisted.
- Call load() to restore values from disk when continuing an experiment.

The configuration behavior is defined in the config package and is used by the population and reproduction logic during evolution.
