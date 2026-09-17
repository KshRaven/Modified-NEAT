import numpy as np

NP_INT = np.int64
NP_FLOAT = np.float32

# ---------------------------------------------------------------------------
# Colors (flat squares only — no gradients/curves anywhere in this game)
# ---------------------------------------------------------------------------
COLOR_BACKGROUND = (16, 16, 18)
COLOR_FLOOR      = (34, 36, 40)      # empty cell
COLOR_WALL       = (215, 215, 220)
COLOR_START      = (60, 110, 200)
COLOR_GOAL       = (60, 200, 110)
COLOR_TRAP       = (200, 60, 60)
COLOR_GRID_LINE  = (24, 25, 28)

COLOR_PLAYER_BOT   = (140, 140, 150)   # ordinary genome
COLOR_PLAYER_RANK1 = (250, 210, 40)    # gold  — highest fitness
COLOR_PLAYER_RANK2 = (210, 210, 220)   # silver
COLOR_PLAYER_RANK3 = (200, 130, 60)    # bronze
COLOR_PLAYER_DEAD  = (90, 45, 45)      # eliminated (drawn briefly, then hidden)

COLOR_PATH_BOT   = (90, 90, 100)
COLOR_PATH_RANK1 = (250, 210, 40)
COLOR_PATH_RANK2 = (210, 210, 220)
COLOR_PATH_RANK3 = (200, 130, 60)

COLOR_TEXT = (230, 230, 230)
