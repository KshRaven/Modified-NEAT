
# ---------- Package that holds all methods required to manipulate genomes ---------- #

from . import initialization, mutation, reproduction, speciation

from .initialization import initialize
from .reproduction import reproduce
from .speciation import speciate
from .mutation import mutate
