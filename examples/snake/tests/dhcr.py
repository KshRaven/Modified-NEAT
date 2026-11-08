"""
Snake AI - Dynamic Hamiltonian Cycle Repair (DHCR), Python reconstruction
of AlphaPhoenix's "Snake_AI_(2020a)_DHCR_with_strategy" (original is MATLAB).

NOTE: written from the video descriptions + public notes, not a line-by-line
port. NacarMode / AppleFearMode are best-effort interpretations.

Cells are ints: id = y*W + x.  Snake deque: [0] = tail ... [-1] = head.
The "cycle" is a successor/predecessor array; the snake body is always a
contiguous segment of it, so following the cycle can never kill the snake.
"""
import random
from collections import deque


# --------------------------------------------------------------------------
# STAGE 1 - game + Hamiltonian cycle
# --------------------------------------------------------------------------
def _zigzag_xy(w, h):
    """Hamiltonian cycle on a w x h grid, h even. Row 0 runs right, the rest
    snakes through columns 1..w-1, column 0 is the return lane."""
    pts = [(x, 0) for x in range(w)]
    for y in range(1, h):
        xs = range(w - 1, 0, -1) if y % 2 else range(1, w)
        pts += [(x, y) for x in xs]
    pts += [(0, y) for y in range(h - 1, 0, -1)]
    return pts


def zigzag_order(W, H):
    if H % 2 == 0:
        return [y * W + x for x, y in _zigzag_xy(W, H)]
    if W % 2 == 0:  # transpose trick
        return [a * W + b for a, b in _zigzag_xy(H, W)]
    raise ValueError("Need at least one even dimension for a Hamiltonian cycle")


class Cycle:
    """Hamiltonian cycle stored as successor / predecessor arrays."""

    def __init__(self, order):
        n = len(order)
        self.succ = [0] * n
        self.pred = [0] * n
        for i, c in enumerate(order):
            nx = order[(i + 1) % n]
            self.succ[c] = nx
            self.pred[nx] = c

    def copy(self):
        c = Cycle.__new__(Cycle)
        c.succ, c.pred = self.succ[:], self.pred[:]
        return c

    def dist(self, a, b):
        """Steps from a to b walking forward along the cycle."""
        d, c = 0, a
        while c != b:
            c = self.succ[c]
            d += 1
        return d


class Dead(Exception):
    pass


class Game:
    def __init__(self, W=30, H=30, seed=None):
        self.W, self.H, self.N = W, H, W * H
        self.rng = random.Random(seed)
        self.nbrs = []
        for c in range(self.N):
            x, y = c % W, c // W
            n = []
            if y > 0: n.append(c - W)
            if x < W - 1: n.append(c + 1)
            if y < H - 1: n.append(c + W)
            if x > 0: n.append(c - 1)
            self.nbrs.append(n)
        self.nset = [set(n) for n in self.nbrs]
        order = zigzag_order(W, H)
        self.cycle = Cycle(order)
        self.snake = deque(order[:3])          # tail ... head, moving along succ
        self.occ = set(self.snake)
        self.steps = 0
        self.apple = self._spawn_apple()

    @property
    def head(self):
        return self.snake[-1]

    @property
    def tail(self):
        return self.snake[0]

    def _spawn_apple(self):
        free = [c for c in range(self.N) if c not in self.occ]
        return self.rng.choice(free) if free else None

    def step(self, nxt):
        """Advance one move. Returns True if an apple was eaten."""
        if nxt not in self.nset[self.head]:
            raise Dead("illegal move")
        ate = nxt == self.apple
        if not ate:
            self.occ.discard(self.snake.popleft())
        if nxt in self.occ:
            raise Dead("hit body")
        self.snake.append(nxt)
        self.occ.add(nxt)
        self.steps += 1
        if ate:
            self.apple = self._spawn_apple()
        return ate

    def won(self):
        return len(self.snake) == self.N


# --------------------------------------------------------------------------
# Baseline agent: pure cycle following (the slow-but-guaranteed method)
# --------------------------------------------------------------------------
class FollowCycleAgent:
    def __init__(self, game):
        self.g = game
        self.cyc = game.cycle

    def choose(self):
        return self.cyc.succ[self.g.head]


def play(game, agent, max_steps=None, on_step=None):
    while not game.won():
        if max_steps and game.steps >= max_steps:
            break
        game.step(agent.choose())
        if on_step:
            on_step(game, agent)
    return game


# --------------------------------------------------------------------------
# STAGE 2 - cycle repair (the "DHCR" part) + direct A*
# --------------------------------------------------------------------------
def repair(cyc, game, h, n, budget=4000):
    """Try to take the shortcut head h -> n while keeping a valid Hamiltonian
    cycle that contains the snake body as one contiguous segment.

    Taking h->n cuts out the path S = succ(h) .. pred(n). S is re-inserted into
    the rest of the cycle in even-length chunks, each spliced into an existing
    cycle edge (u, v) with u~first and last~v (or reversed). Edges between two
    body cells are never touched. Returns a new Cycle, or None if no repair.
    """
    occ, nset, nbrs = game.occ, game.nset, game.nbrs
    if n in occ:
        return None
    succ, pred = cyc.succ[:], cyc.pred[:]
    if succ[h] == n:
        return cyc
    S, c = [], succ[h]
    while c != n:
        if c in occ:
            return None
        S.append(c)
        c = succ[c]
    L = len(S)
    if L % 2:                       # impossible on a bipartite grid; be safe
        return None

    log = []                        # undo log: (array, index, old value)

    def setS(i, v): log.append((succ, i, succ[i])); succ[i] = v
    def setP(i, v): log.append((pred, i, pred[i])); pred[i] = v

    def undo(mark):
        while len(log) > mark:
            arr, i, old = log.pop()
            arr[i] = old

    def inb(x):                     # body cell after the move?
        return x in occ or x == n

    setS(h, n); setP(n, h)
    detached = set(S)
    calls = [budget]

    def options(a, b):
        for u in nbrs[a]:           # u -> a ... b -> v
            if u in detached: continue
            v = succ[u]
            if v in detached or (inb(u) and inb(v)): continue
            if v in nset[b]: yield u, v, False
        for u in nbrs[b]:           # u -> b ... a -> v  (chunk reversed)
            if u in detached: continue
            v = succ[u]
            if v in detached or (inb(u) and inb(v)): continue
            if v in nset[a]: yield u, v, True

    def apply(i, j, u, v, rev):
        a, b = S[i], S[j - 1]
        if not rev:
            setS(u, a); setP(a, u); setS(b, v); setP(v, b)
        else:
            for k in range(i + 1, j):
                setS(S[k], S[k - 1]); setP(S[k - 1], S[k])
            setS(u, b); setP(b, u); setS(a, v); setP(v, a)

    def rec(i):
        if i == L:
            return True
        for j in range(L, i + 1, -2):           # longest chunk first
            a, b = S[i], S[j - 1]
            for u, v, rev in list(options(a, b)):
                calls[0] -= 1
                if calls[0] < 0:
                    return False
                mark = len(log)
                apply(i, j, u, v, rev)
                for k in range(i, j): detached.discard(S[k])
                if rec(j):
                    return True
                undo(mark)
                for k in range(i, j): detached.add(S[k])
        return False

    if not rec(0):
        return None
    out = Cycle.__new__(Cycle)
    out.succ, out.pred = succ, pred
    return out


def verify_cycle(cyc, game):
    """Debug helper: full Hamiltonian cycle + snake contiguous along it."""
    seen, c = set(), 0
    for _ in range(game.N):
        seen.add(c); c = cyc.succ[c]
    assert c == 0 and len(seen) == game.N, "not a Hamiltonian cycle"
    body = list(game.snake)
    for a, b in zip(body, body[1:]):
        assert cyc.succ[a] == b, "snake not contiguous on cycle"


import heapq


def astar_direct(game, goal):
    """Plain A* (Manhattan) head -> goal. Body cells become walkable once the
    tail will have passed them (time-aware). Returns list of cells or None."""
    W = game.W
    start = game.head
    body_idx = {c: i for i, c in enumerate(game.snake)}   # 0 = tail
    gx, gy = goal % W, goal // W
    h0 = lambda c: abs(c % W - gx) + abs(c // W - gy)
    openh = [(h0(start), 0, start)]
    g = {start: 0}
    came = {}
    while openh:
        _, gc, c = heapq.heappop(openh)
        if c == goal:
            path = [c]
            while c in came:
                c = came[c]; path.append(c)
            return path[::-1]
        if gc > g[c]:
            continue
        for n in game.nbrs[c]:
            k = body_idx.get(n)
            if k is not None and k >= gc + 1:       # still occupied on arrival
                continue
            ng = gc + 1
            if ng < g.get(n, 1e9):
                g[n] = ng; came[n] = c
                heapq.heappush(openh, (ng + h0(n), ng, n))
    return None


# --------------------------------------------------------------------------
# STAGE 3 - cycle-aware ("non-Euclidean") A*, candidate ordering, tweaks
# --------------------------------------------------------------------------
from dataclasses import dataclass


@dataclass
class Config:
    off_cycle_cost: float = 0.05     # extra cost for edges not on the cycle
    follow_if_close: int = 0         # follow cycle if its distance <= path len + this
    repair_budget: int = 4000
    max_candidates: int = 3
    # stage 4 hooks (modes)
    nascar: bool = False
    nascar_cost: float = 0.02
    nascar_min_frac: float = 0.0
    fear: bool = False
    fear_cost: float = 0.3
    fear_min_frac: float = 0.0


def astar(game, cyc, goal, cfg):
    """A* over the grid where distance is NOT plain Euclid/Manhattan cost:
    edges that lie on the current Hamiltonian cycle are cheaper than off-cycle
    edges (off-cycle moves need a repair to be legal), so among equal-length
    routes the path hugging the cycle wins. Body cells are time-aware."""
    W = game.W
    succ = cyc.succ
    start = game.head
    body_idx = {c: i for i, c in enumerate(game.snake)}
    gx, gy = goal % W, goal // W
    hf = lambda c: abs(c % W - gx) + abs(c // W - gy)
    frac = len(game.snake) / game.N
    nascar = cfg.nascar and frac >= cfg.nascar_min_frac
    fear = cfg.fear and frac >= cfg.fear_min_frac
    openh = [(hf(start), 0.0, start)]
    g = {start: 0.0}
    steps = {start: 0}
    came = {}
    while openh:
        _, gc, c = heapq.heappop(openh)
        if c == goal:
            path = [c]
            while c in came:
                c = came[c]; path.append(c)
            return path[::-1]
        if gc > g[c]:
            continue
        sc = steps[c]
        p = came.get(c)
        for n in game.nbrs[c]:
            k = body_idx.get(n)
            if k is not None and k >= sc + 1:
                continue
            cost = 1.0
            if succ[c] != n:
                cost += cfg.off_cycle_cost
            if nascar and p is not None:
                cost += _nascar_penalty(W, p, c, n, cfg.nascar_cost)
            if fear and n == goal:
                cost += _fear_penalty(game, cyc, c, n, cfg.fear_cost)
            ng = gc + cost
            if ng < g.get(n, 1e18):
                g[n] = ng; came[n] = c; steps[n] = sc + 1
                heapq.heappush(openh, (ng + hf(n), ng, n))
    return None


# --------------------------------------------------------------------------
# STAGE 4 - strategy modes (interpretations of the video descriptions)
# --------------------------------------------------------------------------
def _nascar_penalty(W, p, c, n, w):
    """NacarMode ("NASCAR"): always turn the same way. The zigzag cycle runs
    clockwise, so left turns are penalised; shortcuts then keep circling like
    a race track instead of carving the free space into thin strips."""
    d1x, d1y = c % W - p % W, c // W - p // W
    d2x, d2y = n % W - c % W, n // W - c // W
    cross = d1x * d2y - d1y * d2x          # >0: right turn (screen coords)
    return w if cross < 0 else 0.0


def _fear_penalty(game, cyc, c, n, w):
    """AppleFearMode: don't swing in toward the apple. The final step onto the
    apple costs extra when it leaves the cycle lane, so the snake approaches
    from the cycle's direction and doesn't wrap tightly around the apple
    (which is how it would trap itself next to its own body)."""
    return w if cyc.succ[c] != n else 0.0


MODES = {
    "plain":   Config(),
    "nascar":  Config(nascar=True),
    "fear":    Config(fear=True),
    "all":     Config(nascar=True, fear=True, nascar_min_frac=0.0,
                      fear_min_frac=0.25, follow_if_close=1),
}


class DHCRAgent:
    def __init__(self, game, cfg=None):
        self.g = game
        self.cfg = cfg or Config()
        self.cyc = game.cycle

    def candidates(self):
        g, cfg = self.g, self.cfg
        h = g.head
        out = []
        path = astar(g, self.cyc, g.apple, cfg)
        if path and len(path) > 1:
            # tweak: if just following the cycle is (nearly) as short, don't
            # perturb the cycle at all
            if cfg.follow_if_close >= 0 and \
               self.cyc.dist(h, g.apple) <= len(path) - 1 + cfg.follow_if_close:
                return []
            out.append(path[1])
        # fallbacks: other free neighbours, closest to the apple first
        W, a = g.W, g.apple
        ax, ay = a % W, a // W
        rest = [n for n in g.nbrs[h] if n not in g.occ and n not in out]
        rest.sort(key=lambda n: abs(n % W - ax) + abs(n // W - ay))
        return out + rest[: max(0, cfg.max_candidates - len(out))]

    def choose(self):
        g, cyc = self.g, self.cyc
        h = g.head
        nxt = cyc.succ[h]
        for n in self.candidates():
            if n == nxt:
                break
            new = repair(cyc, g, h, n, self.cfg.repair_budget)
            if new is not None:
                self.cyc = new
                return n
        return nxt


# --------------------------------------------------------------------------
# STAGE 5 - runner / viewer
# --------------------------------------------------------------------------
def run_gui(game, agent, fps=60, cell=22):
    import pygame
    pygame.init()
    scr = pygame.display.set_mode((game.W * cell, game.H * cell))
    clock = pygame.time.Clock()

    def draw(g, a):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                raise SystemExit
        scr.fill((15, 15, 20))
        n = len(g.snake)
        for i, c in enumerate(g.snake):
            shade = 80 + int(150 * i / max(1, n - 1))
            pygame.draw.rect(scr, (30, shade, 60), (c % g.W * cell + 1, c // g.W * cell + 1, cell - 2, cell - 2))
        if g.apple is not None:
            pygame.draw.rect(scr, (230, 50, 50), (g.apple % g.W * cell + 3, g.apple // g.W * cell + 3, cell - 6, cell - 6))
        pygame.display.set_caption(f"len {n}/{g.N}  steps {g.steps}")
        pygame.display.flip()
        clock.tick(fps)

    play(game, agent, on_step=draw)
    print("won in", game.steps, "steps")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="DHCR Snake AI")
    ap.add_argument("--size", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mode", choices=list(MODES), default="all")
    ap.add_argument("--gui", action="store_true", help="needs pygame")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--bench", action="store_true", help="compare modes (3 seeds)")
    a = ap.parse_args()
    if a.bench:
        print(f"{a.size}x{a.size}, 3 seeds, avg steps to win")
        for name in ("cycle-only", *MODES):
            tot = 0
            for seed in range(3):
                g = Game(a.size, a.size, seed=seed)
                ag = FollowCycleAgent(g) if name == "cycle-only" else DHCRAgent(g, MODES[name])
                play(g, ag); tot += g.steps
            print(f"{name:12s} {tot // 3}")
        return
    g = Game(a.size, a.size, seed=a.seed)
    agent = DHCRAgent(g, MODES[a.mode])
    if a.gui:
        run_gui(g, agent, a.fps)
    else:
        play(g, agent)
        print(f"{a.mode}: won {a.size}x{a.size} in {g.steps} steps")


if __name__ == "__main__":
    main()