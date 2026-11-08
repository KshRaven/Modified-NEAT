"""
Snake AI - CELL-TREE VERSION (Numba-compiled)
=============================================

THE IDEA, IN PLAIN WORDS
------------------------
Group the board into 2x2 blocks ("cells"). A 30x30 board becomes a 15x15 grid
of cells. Now pick a SPANNING TREE over those cells: a set of links between
neighbouring cells that connects all of them with no loops.

Thicken that tree into a corridor system and walk all the way round its
outline. That walk visits every single board square exactly once and returns
to the start: a Hamiltonian cycle (the "ring road"), built from the tree.
Every spanning tree gives one such ring road, and changing ONE tree link
changes the ring road only locally.

Fine-grid rules (cell = 2x2 block, each square has exactly two road links):
  * two squares inside the same cell are linked, UNLESS the tree has a link
    leaving the cell on the side between them (the corridor passes there);
  * two squares in neighbouring cells are linked ONLY IF the tree links those
    two cells.

THE SNAKE
---------
  * Default move: follow the ring road. Always safe: the snake's body is a
    contiguous stretch of the road, so the road ahead is free.
  * Shortcut: to step onto a free neighbour n that is not the next road square,
    change the tree: add the link that makes head->n a road link and remove
    one other link so it stays a tree (a "swap"). The swap is only accepted if
      - the body is STILL one contiguous stretch of the new road, and
      - the new road reaches the apple in fewer moves than the old one.
    So the guarantee never breaks and the snake never dies.
  * Tree edits only touch a few cells, so checking the body is O(1)-ish.

This is a SAFE-BY-CONSTRUCTION cell-tree variant. twanvl's cell-tree
algorithm uses extra heuristic safety conditions to allow more aggressive
tree changes; this one trades a little speed-up for a hard guarantee.

WHY THIS FILE IS FAST
---------------------
Everything (tree, road lookup, tree edits, whole game, many games in parallel
via `prange`) runs inside Numba-compiled code on flat arrays. No dicts, sets or
Python objects in the hot loop.

Board squares are numbered 0..N-1 as  square = y * W + x.  Both W and H must be
even (and >= 4). Cell numbers are  cell = (y//2) * (W//2) + (x//2).
Tree links are stored per cell:  tr[c] = link to the cell on the right,
td[c] = link to the cell below.  Link id = 2*c + (0 right | 1 down).

Usage:
    python cell_tree.py --size 30 --games 4              # play, show deaths
    python cell_tree.py --bench --size 30 --games 100    # benchmark incl. deaths
    python cell_tree.py --selftest                       # internal validity check
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit, prange

__all__ = ["play_many", "play_fast", "pack_config", "MODES_TREE", "bench_deaths"]

# --------------------------------------------------------------------------
# Settings (small integer / float arrays passed to compiled code)
# --------------------------------------------------------------------------
C_EDIT, C_ANY_DIR, C_MAXPATH = 0, 1, 2


def pack_config(edit=True, any_direction=False, max_frac=1.0, stall_factor=2):
    """edit          : allow tree edits (shortcuts). False = plain road following
    any_direction : also try neighbours that are not closer to the apple
    max_frac      : stop editing once the snake fills this fraction of the board
    stall_factor  : if no apple for stall_factor*N moves, just follow the road
    """
    ci = np.zeros(4, np.int64)
    ci[C_EDIT] = int(edit)
    ci[C_ANY_DIR] = int(any_direction)
    ci[3] = stall_factor
    cf = np.array([max_frac], np.float64)
    return ci, cf


MODES_TREE = {
    "cycle": pack_config(edit=False),
    "tree":  pack_config(),
}


# ==========================================================================
# 1. ROAD FROM TREE
# ==========================================================================
@njit(cache=True)
def cadj(s, W, cw, tr, td):
    """The two road neighbours of square s (given the current tree)."""
    x = s % W
    y = s // W
    cx = x >> 1
    cy = y >> 1
    c = cy * cw + cx
    if (y & 1) == 0:                        # top row of its cell
        if cy > 0 and td[c - cw] == 1:
            a = (y - 1) * W + x
        else:
            a = y * W + (x ^ 1)
    else:                                   # bottom row
        if td[c] == 1:
            a = (y + 1) * W + x
        else:
            a = y * W + (x ^ 1)
    if (x & 1) == 0:                        # left column
        if cx > 0 and tr[c - 1] == 1:
            b = y * W + x - 1
        else:
            b = (y ^ 1) * W + x
    else:                                   # right column
        if tr[c] == 1:
            b = y * W + x + 1
        else:
            b = (y ^ 1) * W + x
    return a, b


@njit(cache=True)
def road_next(cur, prev, W, cw, tr, td):
    a, b = cadj(cur, W, cw, tr, td)
    if a == prev:
        return b
    return a


@njit(cache=True)
def walk_dist(start, prev, target, N, W, cw, tr, td):
    """Moves from `start` to `target`, going along the road away from `prev`."""
    d = 0
    cur = start
    p = prev
    while cur != target:
        nx = road_next(cur, p, W, cw, tr, td)
        p = cur
        cur = nx
        d += 1
        if d > N:
            return -1
    return d


@njit(cache=True)
def set_edge(e, val, tr, td):
    c = e >> 1
    if (e & 1) == 1:
        td[c] = val
    else:
        tr[c] = val


@njit(cache=True)
def has_edge(e, tr, td):
    c = e >> 1
    if (e & 1) == 1:
        return td[c] == 1
    return tr[c] == 1


@njit(cache=True)
def edge_id(c, d, cw):
    """Link id between neighbouring cells c and d."""
    if d == c + 1:
        return c * 2
    if d == c - 1:
        return d * 2
    if d == c + cw:
        return c * 2 + 1
    return d * 2 + 1


@njit(cache=True)
def edge_ends(e, cw):
    c = e >> 1
    if (e & 1) == 1:
        return c, c + cw
    return c, c + 1


@njit(cache=True)
def tree_nbrs(c, cw, tr, td, out):
    cx = c % cw
    cy = c // cw
    k = 0
    if tr[c] == 1:
        out[k] = c + 1; k += 1
    if cx > 0 and tr[c - 1] == 1:
        out[k] = c - 1; k += 1
    if td[c] == 1:
        out[k] = c + cw; k += 1
    if cy > 0 and td[c - cw] == 1:
        out[k] = c - cw; k += 1
    return k


@njit(cache=True)
def init_tree(cw, ch, tr, td):
    """Comb-shaped starting tree: top row linked sideways, every column
    linked downwards."""
    for cx in range(cw - 1):
        tr[cx] = 1
    for c in range(cw * (ch - 1)):
        td[c] = 1


@njit(cache=True)
def tree_path(A, B, cw, tr, td, qu, pc, vis, gen, nb, path):
    """Links on the tree path from cell A to cell B. Returns how many."""
    head = 0
    tail = 1
    qu[0] = A
    vis[A] = gen
    while head < tail:
        c = qu[head]
        head += 1
        if c == B:
            break
        k = tree_nbrs(c, cw, tr, td, nb)
        for i in range(k):
            d = nb[i]
            if vis[d] != gen:
                vis[d] = gen
                pc[d] = c
                qu[tail] = d
                tail += 1
    if vis[B] != gen:
        return -1
    n = 0
    c = B
    while c != A:
        p = pc[c]
        path[n] = edge_id(p, c, cw)
        n += 1
        c = p
    return n


@njit(cache=True)
def tree_mark(A, cw, tr, td, qu, vis, gen, nb):
    """Mark (vis == gen) every cell reachable from A in the tree."""
    head = 0
    tail = 1
    qu[0] = A
    vis[A] = gen
    while head < tail:
        c = qu[head]
        head += 1
        k = tree_nbrs(c, cw, tr, td, nb)
        for i in range(k):
            d = nb[i]
            if vis[d] != gen:
                vis[d] = gen
                qu[tail] = d
                tail += 1


# ==========================================================================
# 2. TREE EDITS: is a swap legal, and how good is it?
# ==========================================================================
@njit(cache=True)
def eval_edit(h, n, W, N, cw, tr, td, occ, pos, ring, tail_ptr, L, apple, cells):
    """The tree has ALREADY been edited. Return the number of moves from h to
    the apple along the new road (via n), or -1 if the edit is not allowed.

    Allowed = h->n is a road link AND every body link around the changed cells
    is still a road link (so the body is still one contiguous stretch)."""
    a, b = cadj(h, W, cw, tr, td)
    if a != n and b != n:
        return -1
    for z in range(4):
        cc = cells[z]
        cx = cc % cw
        cy = cc // cw
        for q in range(4):
            s = (cy * 2 + (q >> 1)) * W + cx * 2 + (q & 1)
            if not occ[s]:
                continue
            ps = pos[s]
            k = ps - tail_ptr
            if k < 0:
                k += N
            if k > 0:
                t = ring[(ps + N - 1) % N]
                a, b = cadj(s, W, cw, tr, td)
                if a != t and b != t:
                    return -1
            if k < L - 1:
                t = ring[(ps + 1) % N]
                a, b = cadj(s, W, cw, tr, td)
                if a != t and b != t:
                    return -1
    d2 = walk_dist(n, h, apple, N, W, cw, tr, td)
    if d2 < 0:
        return -1
    return d2 + 1


@njit(cache=True)
def try_candidate(h, n, W, H, N, cw, ch, tr, td, occ, pos, ring, tail_ptr, L,
                  apple, best_d, work):
    """Look for the best tree swap that makes h->n a road link and brings the
    apple closer than best_d. Returns (new best_d, link to remove, link to add);
    remove = -1 if nothing better was found."""
    qu, pc, vis, nb, path, cells, gen = work
    best_rem = -1
    best_add = -1
    cA = (h // W // 2) * cw + (h % W) // 2
    cB = (n // W // 2) * cw + (n % W) // 2
    gen[0] += 1
    if cA != cB:
        # h->n crosses between two cells: that needs a tree link A-B.
        add = edge_id(cA, cB, cw)
        if has_edge(add, tr, td):
            return best_d, -1, -1
        m = tree_path(cA, cB, cw, tr, td, qu, pc, vis, gen[0], nb, path)
        for i in range(m):
            e = path[i]
            set_edge(e, 0, tr, td)
            set_edge(add, 1, tr, td)
            e1, e2 = edge_ends(e, cw)
            a1, a2 = edge_ends(add, cw)
            cells[0] = e1; cells[1] = e2; cells[2] = a1; cells[3] = a2
            d = eval_edit(h, n, W, N, cw, tr, td, occ, pos, ring, tail_ptr, L,
                          apple, cells)
            set_edge(add, 0, tr, td)
            set_edge(e, 1, tr, td)
            if d >= 0 and d < best_d:
                best_d = d
                best_rem = e
                best_add = add
        return best_d, best_rem, best_add
    # Same cell: the step is blocked by a tree link leaving that side.
    hx = h % W
    hy = h // W
    if n == h + 1 or n == h - 1:            # sideways step
        if (hy & 1) == 0:
            if hy < 2:
                return best_d, -1, -1
            e = (cA - cw) * 2 + 1           # link up
        else:
            e = cA * 2 + 1                  # link down
    else:                                   # vertical step
        if (hx & 1) == 0:
            if hx < 2:
                return best_d, -1, -1
            e = (cA - 1) * 2                # link left
        else:
            e = cA * 2                      # link right
    if not has_edge(e, tr, td):
        return best_d, -1, -1               # already a road link
    set_edge(e, 0, tr, td)                  # remove it: tree splits in two
    tree_mark(cA, cw, tr, td, qu, vis, gen[0], nb)
    for c in range(cw * ch):                # reconnect with any other link
        cx = c % cw
        cy = c // cw
        for dr in range(2):
            if dr == 0:
                if cx == cw - 1:
                    continue
                d2c = c + 1
            else:
                if cy == ch - 1:
                    continue
                d2c = c + cw
            g = c * 2 + dr
            if g == e or has_edge(g, tr, td):
                continue
            if (vis[c] == gen[0]) == (vis[d2c] == gen[0]):
                continue                    # same side: would make a loop
            set_edge(g, 1, tr, td)
            e1, e2 = edge_ends(e, cw)
            cells[0] = e1; cells[1] = e2; cells[2] = c; cells[3] = d2c
            d = eval_edit(h, n, W, N, cw, tr, td, occ, pos, ring, tail_ptr, L,
                          apple, cells)
            set_edge(g, 0, tr, td)
            if d >= 0 and d < best_d:
                best_d = d
                best_rem = e
                best_add = g
    set_edge(e, 1, tr, td)
    return best_d, best_rem, best_add


@njit(cache=True)
def choose_move(W, H, N, cw, ch, nbrs, tr, td, occ, pos, ring, tail_ptr, L,
                apple, cfg, cfgf, work):
    """Pick the next square (and possibly edit the tree)."""
    h = ring[(tail_ptr + L - 1) % N]
    p2 = ring[(tail_ptr + L - 2) % N]
    nxt = road_next(h, p2, W, cw, tr, td)
    if cfg[C_EDIT] == 0 or L >= cfgf[0] * N:
        return nxt
    d = walk_dist(h, p2, apple, N, W, cw, tr, td)
    ax = apple % W
    ay = apple // W
    man = abs(h % W - ax) + abs(h // W - ay)
    if d <= man:                            # road is already optimal
        return nxt
    best_d = d
    best_n = -1
    best_rem = -1
    best_add = -1
    for t in range(4):
        n = nbrs[h, t]
        if n < 0 or occ[n] or n == nxt:
            continue
        if cfg[C_ANY_DIR] == 0 and abs(n % W - ax) + abs(n // W - ay) >= man:
            continue
        bd, rem, add = try_candidate(h, n, W, H, N, cw, ch, tr, td, occ, pos,
                                     ring, tail_ptr, L, apple, best_d, work)
        if rem >= 0:
            best_d = bd
            best_n = n
            best_rem = rem
            best_add = add
    if best_n >= 0:
        set_edge(best_rem, 0, tr, td)       # commit the best swap
        set_edge(best_add, 1, tr, td)
        return best_n
    return nxt


# ==========================================================================
# 3. WHOLE GAME INSIDE COMPILED CODE
# ==========================================================================
@njit(cache=True)
def _rand(state):
    x = state[0]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    state[0] = x
    return x


@njit(cache=True)
def _spawn_apple(N, occ, rng, nfree):
    if nfree <= 0:
        return -1
    k = int(_rand(rng) % np.uint64(nfree))
    for c in range(N):
        if not occ[c]:
            if k == 0:
                return c
            k -= 1
    return -1


@njit(cache=True)
def build_nbrs(W, H):
    nbrs = np.full((W * H, 4), -1, dtype=np.int32)
    for c in range(W * H):
        x, y = c % W, c // W
        if y > 0:     nbrs[c, 0] = c - W
        if x < W - 1: nbrs[c, 1] = c + 1
        if y < H - 1: nbrs[c, 2] = c + W
        if x > 0:     nbrs[c, 3] = c - 1
    return nbrs


@njit(cache=True)
def is_hamiltonian(W, N, cw, tr, td):
    """Debug: does the current tree give one road through all N squares?"""
    cur = 0
    prev = cadj(0, W, cw, tr, td)[0]
    for _ in range(N):
        nx = road_next(cur, prev, W, cw, tr, td)
        prev = cur
        cur = nx
        if cur == 0:
            return _ == N - 1
    return False


@njit(cache=True)
def run_game(W, H, seed, cfg, cfgf, max_steps, debug):
    """Play one game. Returns (moves, final length, status).
    status: 0 won, 1 died (should never happen), 2 move cap,
            3 internal road check failed (debug only)."""
    N = W * H
    cw = W // 2
    ch = H // 2
    nbrs = build_nbrs(W, H)
    tr = np.zeros(cw * ch, np.uint8)
    td = np.zeros(cw * ch, np.uint8)
    init_tree(cw, ch, tr, td)
    work = (np.zeros(cw * ch, np.int32), np.zeros(cw * ch, np.int32),
            np.zeros(cw * ch, np.int32), np.zeros(4, np.int32),
            np.zeros(cw * ch, np.int32), np.zeros(4, np.int32),
            np.zeros(1, np.int32))
    occ = np.zeros(N, np.uint8)
    pos = np.zeros(N, np.int32)
    ring = np.zeros(N, np.int32)
    # start: three squares along the road from square 0
    c0 = 0
    c1 = cadj(0, W, cw, tr, td)[1]
    c2 = road_next(c1, c0, W, cw, tr, td)
    ring[0] = c0; ring[1] = c1; ring[2] = c2
    for i in range(3):
        pos[ring[i]] = i
        occ[ring[i]] = 1
    tail_ptr = 0
    L = 3
    rng = np.zeros(1, np.uint64)
    rng[0] = np.uint64(seed) * np.uint64(2654435761) + np.uint64(88172645463325252)
    for _ in range(4):
        _rand(rng)
    apple = _spawn_apple(N, occ, rng, N - L)
    moves = 0
    since = 0
    status = 2
    while L < N and moves < max_steps:
        h = ring[(tail_ptr + L - 1) % N]
        if since > cfg[3] * N:              # stall guard: plain road
            p2 = ring[(tail_ptr + L - 2) % N]
            nxt = road_next(h, p2, W, cw, tr, td)
        else:
            nxt = choose_move(W, H, N, cw, ch, nbrs, tr, td, occ, pos, ring,
                              tail_ptr, L, apple, cfg, cfgf, work)
        since += 1
        ate = nxt == apple
        if not ate:
            occ[ring[tail_ptr]] = 0
            tail_ptr = (tail_ptr + 1) % N
            L -= 1
        if occ[nxt]:
            status = 1
            break
        slot = (tail_ptr + L) % N
        ring[slot] = nxt
        pos[nxt] = slot
        occ[nxt] = 1
        L += 1
        moves += 1
        if debug and moves % 7 == 0:
            if not is_hamiltonian(W, N, cw, tr, td):
                status = 3
                break
        if ate:
            since = 0
            apple = _spawn_apple(N, occ, rng, N - L)
    if L == N:
        status = 0
    return moves, L, status


@njit(parallel=True, cache=True)
def play_batch(W, H, seeds, cfg, cfgf, max_steps, debug, out_steps, out_len,
               out_status):
    """Play many independent games at once, one per CPU core."""
    for i in prange(seeds.shape[0]):
        m, L, s = run_game(W, H, seeds[i], cfg, cfgf, max_steps, debug)
        out_steps[i] = m
        out_len[i] = L
        out_status[i] = s


# ==========================================================================
# 4. FRIENDLY WRAPPERS, BENCHMARK, CLI
# ==========================================================================
def _cfg(mode):
    return MODES_TREE[mode] if isinstance(mode, str) else mode


def _check_size(W, H):
    if W % 2 or H % 2 or W < 4 or H < 4:
        raise ValueError("cell_tree needs even width and height >= 4")


def play_many(W, H, seeds, mode="tree", max_steps=0, debug=False):
    """One game per seed, in parallel. Returns (moves, lengths, statuses)."""
    _check_size(W, H)
    seeds = np.asarray(seeds, dtype=np.int64)
    ci, cf = _cfg(mode)
    cap = max_steps or (W * H) ** 2 * 20
    o1 = np.zeros(len(seeds), np.int64)
    o2 = np.zeros(len(seeds), np.int64)
    o3 = np.zeros(len(seeds), np.int64)
    play_batch(W, H, seeds, ci, cf, cap, debug, o1, o2, o3)
    return o1, o2, o3


def play_fast(W, H=None, seed=0, mode="tree", max_steps=0):
    H = H or W
    _check_size(W, H)
    ci, cf = _cfg(mode)
    m, L, s = run_game(W, H, seed, ci, cf, max_steps or (W * H) ** 2 * 20, False)
    return {"moves": int(m), "length": int(L), "won": s == 0, "status": int(s)}


BENCH_MODES = ("cycle", "tree")


def bench_deaths(size=30, games=100, modes=BENCH_MODES, seed0=0, max_steps=0):
    """Play `games` games per mode and report deaths as well as move counts.
    Move statistics are over WON games only."""
    play_many(4, 4, np.arange(1), "tree")           # compile / load cache
    seeds = np.arange(seed0, seed0 + games)
    print(f"{size}x{size}, {games} games per mode, seeds {seed0}..{seed0 + games - 1}")
    hdr = (f"{'mode':<10}{'wins':>6}{'deaths':>8}{'death %':>9}{'capped':>8}"
           f"{'mean moves':>12}{'median':>10}{'std':>9}{'min':>9}{'max':>9}{'time':>8}")
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for m in modes:
        t0 = time.perf_counter()
        mv, ln, st = play_many(size, size, seeds, m, max_steps)
        dt = time.perf_counter() - t0
        won = st == 0
        d = int((st == 1).sum())
        c = int((st == 2).sum())
        w = mv[won]
        if len(w):
            stats = (f"{w.mean():>12.0f}{np.median(w):>10.0f}{w.std():>9.0f}"
                     f"{w.min():>9d}{w.max():>9d}")
        else:
            stats = f"{'-':>12}{'-':>10}{'-':>9}{'-':>9}{'-':>9}"
        print(f"{m:<10}{int(won.sum()):>6}{d:>8}{100.0 * d / games:>8.1f}%{c:>8}"
              f"{stats}{dt:>7.1f}s")
        rows[m] = dict(wins=int(won.sum()), deaths=d, capped=c,
                       mean=float(w.mean()) if len(w) else None)
    return rows


def selftest():
    """Play small games with the full road check switched on."""
    for size in (4, 6, 8, 10, 12):
        mv, ln, st = play_many(size, size, np.arange(6), "tree", debug=True)
        bad = int((st == 3).sum())
        print(f"{size}x{size}: moves {int(mv.mean())}, deaths {int((st == 1).sum())}, "
              f"road-check failures {bad}, all won {bool((st == 0).all())}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Numba cell-tree Snake AI")
    ap.add_argument("--size", type=int, default=30)
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--mode", choices=list(MODES_TREE), default="tree")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true",
                    help="benchmark incl. deaths (--games, default 100 here)")
    ap.add_argument("--modes", nargs="+", choices=list(MODES_TREE))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.bench:
        bench_deaths(a.size, a.games if a.games > 1 else 100,
                     a.modes or BENCH_MODES, a.seed)
    else:
        seeds = np.arange(a.seed, a.seed + a.games)
        play_many(4, 4, seeds[:1], a.mode)          # warm-up / compile
        t0 = time.perf_counter()
        mv, ln, st = play_many(a.size, a.size, seeds, a.mode)
        dt = time.perf_counter() - t0
        for s_, m, k in zip(seeds, mv, st):
            tag = {0: "won", 1: "DIED", 2: "capped"}.get(int(k), "ERROR")
            print(f"seed {s_}: {tag} {a.size}x{a.size} in {m} moves")
        print(f"{a.games} game(s) in {dt:.3f}s, deaths: {int((st == 1).sum())}")