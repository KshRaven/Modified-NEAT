from numba import types, typeof, optional, objmode
from numba.experimental import jitclass
from numba.typed import Dict

FLOAT   = types.float64
INT     = types.int64
STR     = typeof('str')
BOOL    = types.boolean


@jitclass([
    ('key', INT),
    ('fitness', optional(FLOAT)),
    ('_actual', optional(FLOAT)),
    ('genus', INT)
])
class Genome(object):
    """
    Identity and fitness of an individual. Its parameters (weights/biases) live in the NeatModule tensors,
    indexed through ``NeatModule.mapping[genome.key]``; crossover, mutation, distance and initialization run in
    ``ModifiedNEAT.base`` (CPU / GPU backends).
    """
    def __init__(self, key: int, genus: int):
        self.key = key
        self.fitness: float | None = None
        self._actual: float | None = None
        self.genus = genus

    @property
    def actual_fitness(self):
        return self.fitness if self._actual is None else self._actual

    def __str__(self):
        with objmode(res='unicode_type'):
            res = (
                "Genome(key="+str(self.key)+", genus="+str(self.genus)+", "
                "fitness="+str(self.fitness)+")"
            )
        return res


def load_genome(struct: Dict):
    key             = struct['key']
    genus           = struct['genus']
    genome          = Genome(key, genus)
    genome.fitness  = struct['fitness']
    return genome
