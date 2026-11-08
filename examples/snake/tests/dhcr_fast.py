"""
Snake AI (Dynamic Hamiltonian Cycle Repair) - NUMBA-COMPILED VERSION
=====================================================================

THE IDEA, IN PLAIN WORDS
------------------------
Picture the game board as a city and draw ONE one-way ring road that passes
through every single square exactly once and ends back where it started.
That ring road is called a "Hamiltonian cycle".

  * If the snake only ever drives along the ring road, it can never crash into
    itself: its own body is just a stretch of the road it already drove, and
    the road ahead is always free. It will always win... but slowly, because
    it drives past the apple and keeps looping around the whole board.

  * So we let the snake take SHORTCUTS: cut across to a neighbouring square
    that is further along the road, which skips a chunk of road.
    Skipping road is only safe if every skipped square is still reachable
    later. So after each shortcut we REPAIR the ring road: the skipped squares
    are stitched back into the road somewhere else, so that there is again one
    complete ring road through every square, with the snake's body sitting on
    it in one piece. (Think: re-routing a train line around a detour so the
    loop stays unbroken.) If no repair exists, we simply don't take that
    shortcut. The ring-road guarantee is never lost, so the snake never dies.

  * To decide WHICH shortcut, we ask a route-finder (A*, like a GPS) for the
    shortest way to the apple, with two small nudges:
       - using the ring road costs a tiny bit less than leaving it, because
         leaving it requires a repair;
       - the GPS knows the snake's tail moves away each turn, so squares the
         tail is about to leave count as free.

  * Two "driving style" modes (our best reading of the original video):
       NASCAR mode      - always prefer turning the same way, like racing
                          round an oval; this keeps the free space in big,
                          tidy pieces instead of thin strips.
       APPLE-FEAR mode  - don't swing in sideways toward the apple at the last
                          step; arrive along the road instead, so the snake
                          doesn't wrap itself tightly around the apple.

A HONEST NOTE ON HOW GOOD THE SHORTCUTS ARE
-------------------------------------------
This is a reconstruction, and the "repair" step is strict: in a 20x20 game the
route-finder proposes a shortcut roughly 18,000 times but only about 4% of
them can be stitched back into a valid ring road. So the snake finishes a
30x30 board in about the same number of moves as plain road-following
(~200k). The compiled version is much FASTER at playing, but it makes the
same moves the Python version would; it does not make the snake smarter.
A richer repair (e.g. re-ordering skipped pieces, not just re-inserting them)
is the place to look for big gains in moves.

WHY THIS FILE IS FAST
---------------------
Plain Python re-reads and re-interprets every line each time it runs, and its
dictionaries / sets / lists are flexible but slow. Numba instead translates
the functions below into real machine code the first time they run (a few
seconds, then cached on disk), and that code works on plain number arrays
like a C program would. Concretely we:

  1. keep everything in flat arrays (road, snake, free squares) - no dicts/sets;
  2. write our own tiny "priority queue" for the GPS instead of Python's heapq;
  3. repair the road IN PLACE and keep an undo list, instead of copying the
     whole road for every shortcut we try;
  4. run the WHOLE game (move choosing, snake movement, apple placing) inside
     one compiled call, so Python is never involved per step;
  5. play many independent games at once, one per CPU core (`prange`).

Style follows the env's `kernels/` package: `@njit(parallel=..., fastmath=True,
cache=True)`, flat arrays, in-place updates, `prange` over independent items.

Board squares are numbered 0..N-1 as  square = y * W + x.
"""
from __future__ import annotations

import sys
import time

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

__all__ = ["play_fast", "play_many", "pack_config", "MODES_FAST"]

# --------------------------------------------------------------------------
# Settings are passed to the compiled code as a small array of numbers.
# These are just the positions ("slots") inside that array.
# Costs are whole numbers x1000 (1.0 step = 1000) so comparisons are exact.
# --------------------------------------------------------------------------
C_OFF, C_NASCAR, C_NASCAR_COST, C_FEAR, C_FEAR_COST = 0, 1, 2, 3, 4
C_FOLLOW, C_BUDGET, C_MAXCAND = 5, 6, 7


def pack_config(off_cycle_cost=0.05, follow_if_close=0, repair_budget=4000,
                max_candidates=3, nascar=False, nascar_cost=0.02,
                nascar_min_frac=0.0, fear=False, fear_cost=0.3,
                fear_min_frac=0.0):
    """Turn friendly settings into the two small arrays the kernels read."""
    ci = np.zeros(8, dtype=np.int64)
    ci[C_OFF] = round(off_cycle_cost * 1000)
    ci[C_NASCAR] = int(nascar)
    ci[C_NASCAR_COST] = round(nascar_cost * 1000)
    ci[C_FEAR] = int(fear)
    ci[C_FEAR_COST] = round(fear_cost * 1000)
    ci[C_FOLLOW] = follow_if_close
    ci[C_BUDGET] = repair_budget
    ci[C_MAXCAND] = max_candidates
    cf = np.array([nascar_min_frac, fear_min_frac], dtype=np.float64)
    return ci, cf


# Same four presets as the pure-Python version.
MODES_FAST = {
    "plain":  pack_config(),
    "nascar": pack_config(nascar=True),
    "fear":   pack_config(fear=True),
    "all":    pack_config(nascar=True, fear=True, fear_min_frac=0.25,
                          follow_if_close=1),
}


# ==========================================================================
# 1. BOARD SET-UP
# ==========================================================================
@njit(cache=True)
def build_nbrs(W: int, H: int) -> Array:
    """For every square, list its up/right/down/left neighbours (-1 = wall)."""
    nbrs = np.full((W * H, 4), -1, dtype=np.int32)
    for c in range(W * H):
        x, y = c % W, c // W
        if y > 0:     nbrs[c, 0] = c - W
        if x < W - 1: nbrs[c, 1] = c + 1
        if y < H - 1: nbrs[c, 2] = c + W
        if x > 0:     nbrs[c, 3] = c - 1
    return nbrs


@njit(cache=True)
def build_cycle(W: int, H: int, succ: Array, pred: Array) -> Array:
    """Draw the first ring road: a simple back-and-forth ("zigzag") sweep.

    Along the top row we go right; then we sweep left/right through the rest of
    the board leaving the leftmost column free; finally we come back up that
    free column to the start. `succ[c]` = the square after c on the road,
    `pred[c]` = the square before it. Returns the squares in road order.
    """
    N = W * H
    order = np.empty(N, dtype=np.int32)
    k = 0
    if H % 2 == 0:
        w, h = W, H
        swap = False
    else:                                   # need an even side: flip the board
        w, h = H, W
        swap = True
    for x in range(w):                      # top row, left -> right
        order[k] = (x * W) if swap else x
        k += 1
    for y in range(1, h):                   # zigzag through columns 1..w-1
        if y % 2 == 1:
            for x in range(w - 1, 0, -1):
                order[k] = (x * W + y) if swap else (y * W + x); k += 1
        else:
            for x in range(1, w):
                order[k] = (x * W + y) if swap else (y * W + x); k += 1
    for y in range(h - 1, 0, -1):           # back up the free left column
        order[k] = (0 * W + y) if swap else (y * W + 0); k += 1
    for i in range(N):
        a = order[i]
        b = order[(i + 1) % N]
        succ[a] = b
        pred[b] = a
    return order


@njit(cache=True)
def make_work(N: int):
    """Scratch memory, allocated ONCE per game and reused every move."""
    D = N // 2 + 2                          # max nesting of repair attempts
    return (
        np.zeros(N, np.int64),              # 0  gcost : best known route cost
        np.zeros(N, np.int32),              # 1  stamp : "is gcost valid this search?"
        np.zeros(N, np.int32),              # 2  steps : moves taken to get there
        np.zeros(N, np.int32),              # 3  came  : square we arrived from
        np.zeros(4 * N + 8, np.int64),      # 4  heap f
        np.zeros(4 * N + 8, np.int64),      # 5  heap g
        np.zeros(4 * N + 8, np.int32),      # 6  heap node
        np.zeros(N, np.int32),              # 7  S : squares a shortcut skips
        np.zeros(N, np.uint8),              # 8  det : "skipped & not yet re-added"
        np.zeros(8 * N, np.uint8),          # 9  undo log: which road array
        np.zeros(8 * N, np.int32),          # 10 undo log: index
        np.zeros(8 * N, np.int32),          # 11 undo log: old value
        np.zeros(D, np.int32),              # 12 frame: chunk start
        np.zeros(D, np.int32),              # 13 frame: chunk end
        np.zeros(D, np.int32),              # 14 frame: next option to try
        np.zeros(D, np.int32),              # 15 frame: number of options
        np.zeros(D, np.int32),              # 16 frame: undo-log mark
        np.zeros((D, 8, 3), np.int32),      # 17 frame: the options themselves
    )


# ==========================================================================
# 2. THE ROUTE-FINDER ("GPS" = A*)
# ==========================================================================
@njit(cache=True)
def _less(f1, g1, n1, f2, g2, n2):
    """Heap ordering: smaller estimate first; ties -> cheaper so far -> lower id."""
    if f1 != f2:
        return f1 < f2
    if g1 != g2:
        return g1 < g2
    return n1 < n2


@njit(cache=True)
def _heap_push(hf, hg, hn, size, f, g, n):
    """Add a candidate to the 'to-do list', keeping the best one on top."""
    i = size
    hf[i] = f; hg[i] = g; hn[i] = n
    while i > 0:
        p = (i - 1) >> 1
        if _less(hf[i], hg[i], hn[i], hf[p], hg[p], hn[p]):
            hf[i], hf[p] = hf[p], hf[i]
            hg[i], hg[p] = hg[p], hg[i]
            hn[i], hn[p] = hn[p], hn[i]
            i = p
        else:
            break
    return size + 1


@njit(cache=True)
def _heap_pop(hf, hg, hn, size):
    """Take the best candidate off the top of the to-do list."""
    f = hf[0]; g = hg[0]; n = hn[0]
    size -= 1
    if size > 0:
        hf[0] = hf[size]; hg[0] = hg[size]; hn[0] = hn[size]
        i = 0
        while True:
            l = 2 * i + 1
            r = l + 1
            m = i
            if l < size and _less(hf[l], hg[l], hn[l], hf[m], hg[m], hn[m]):
                m = l
            if r < size and _less(hf[r], hg[r], hn[r], hf[m], hg[m], hn[m]):
                m = r
            if m == i:
                break
            hf[i], hf[m] = hf[m], hf[i]
            hg[i], hg[m] = hg[m], hg[i]
            hn[i], hn[m] = hn[m], hn[i]
            i = m
    return f, g, n, size


@njit(cache=True, fastmath=True)
def astar_first(head, goal, W, N, nbrs, succ, occ, pos, tail_ptr,
                nascar_on, fear_on, off_cost, nas_cost, fear_cost,
                gcost, stamp, gen, steps, came, hf, hg, hn):
    """Shortest route from the snake's head to the apple.

    In plain words: explore outward from the head, always extending the most
    promising route first (cost so far + straight-line guess of what's left).
    Each step costs 1.0, plus small nudges:
      * +off_cost    if the step leaves the ring road (needs a repair);
      * +nas_cost    if the step is a LEFT turn (NASCAR mode: keep turning right);
      * +fear_cost   if the very last step onto the apple leaves the road
                     (APPLE-FEAR mode).
    A body square counts as blocked only if the tail will NOT have moved off it
    by the time we'd arrive (the tail advances one square per move).

    Returns (first square to step to, route length in moves), or (-1, -1).
    """
    gx = goal % W
    gy = goal // W
    size = 0
    gcost[head] = 0
    stamp[head] = gen
    steps[head] = 0
    h0 = 1000 * (abs(head % W - gx) + abs(head // W - gy))
    size = _heap_push(hf, hg, hn, size, h0, 0, head)
    while size > 0:
        f, gc, c, size = _heap_pop(hf, hg, hn, size)
        if c == goal:
            node = goal
            first = -1
            length = 0
            while node != head:
                first = node
                node = came[node]
                length += 1
            return first, length
        if stamp[c] == gen and gc > gcost[c]:
            continue                                   # stale list entry
        sc = steps[c]
        for t in range(4):
            n = nbrs[c, t]
            if n < 0:
                continue
            if occ[n]:
                k = pos[n] - tail_ptr                   # distance from tail
                if k < 0:
                    k += N
                if k >= sc + 1:                         # tail hasn't left yet
                    continue
            cost = 1000
            if succ[c] != n:
                cost += off_cost
            if nascar_on and c != head:
                p = came[c]
                d1x = c % W - p % W
                d1y = c // W - p // W
                d2x = n % W - c % W
                d2y = n // W - c // W
                if d1x * d2y - d1y * d2x < 0:           # left turn
                    cost += nas_cost
            if fear_on and n == goal and succ[c] != n:
                cost += fear_cost
            ng = gc + cost
            if stamp[n] != gen or ng < gcost[n]:
                stamp[n] = gen
                gcost[n] = ng
                came[n] = c
                steps[n] = sc + 1
                est = ng + 1000 * (abs(n % W - gx) + abs(n // W - gy))
                size = _heap_push(hf, hg, hn, size, est, ng, n)
    return -1, -1


# ==========================================================================
# 3. THE ROAD REPAIR  (the clever part)
# ==========================================================================
@njit(cache=True)
def _ws(succ, pred, la, li, lo, nlog, which, idx, val):
    """Change one road link, and remember the old value so we can undo it."""
    la[nlog] = which
    li[nlog] = idx
    if which == 0:
        lo[nlog] = succ[idx]; succ[idx] = val
    else:
        lo[nlog] = pred[idx]; pred[idx] = val
    return nlog + 1


@njit(cache=True)
def _undo(succ, pred, la, li, lo, nlog, mark):
    """Roll the road back to how it looked when the log had `mark` entries."""
    while nlog > mark:
        nlog -= 1
        if la[nlog] == 0:
            succ[li[nlog]] = lo[nlog]
        else:
            pred[li[nlog]] = lo[nlog]
    return nlog


@njit(cache=True)
def _gen_options(a, b, n, W, nbrs, succ, occ, det, opts, f):
    """Find places to re-attach a skipped stretch of road running a..b.

    A stretch can be spliced into an existing road link u->v if its two ends
    touch u and v (either way round). We never touch links that belong to the
    snake's body. Returns how many options were written into opts[f].
    """
    cnt = 0
    bx = b % W; by = b // W
    ax = a % W; ay = a // W
    for t in range(4):                      # u -> a ... b -> v
        u = nbrs[a, t]
        if u < 0 or det[u]:
            continue
        v = succ[u]
        if det[v]:
            continue
        if (occ[u] or u == n) and (occ[v] or v == n):
            continue
        if abs(v % W - bx) + abs(v // W - by) == 1:
            opts[f, cnt, 0] = u; opts[f, cnt, 1] = v; opts[f, cnt, 2] = 0
            cnt += 1
    for t in range(4):                      # u -> b ... a -> v (stretch flipped)
        u = nbrs[b, t]
        if u < 0 or det[u]:
            continue
        v = succ[u]
        if det[v]:
            continue
        if (occ[u] or u == n) and (occ[v] or v == n):
            continue
        if abs(v % W - ax) + abs(v // W - ay) == 1:
            opts[f, cnt, 0] = u; opts[f, cnt, 1] = v; opts[f, cnt, 2] = 1
            cnt += 1
    return cnt


@njit(cache=True, fastmath=True)
def repair(h, n, W, N, nbrs, succ, pred, occ, budget, work):
    """Try to take the shortcut head h -> n and keep a valid ring road.

    In plain words:
      1. Taking h->n skips the stretch of road between them (call it S).
      2. Cut S into pieces and glue each piece back into the road somewhere it
         fits (its two ends must touch the two ends of an existing road link).
      3. If we get stuck, undo the last glue and try another way (a search with
         backtracking). A work budget stops it from searching forever.
      4. If everything fits: the road is again one full loop, snake body intact,
         and we return True. Otherwise everything is undone and we return False.
    Works directly on `succ`/`pred` (no copies) using an undo log.
    """
    (_, _, _, _, _, _, _, S, det, la, li, lo,
     fi, fj, fk, fn_, fmark, opts) = work
    if occ[n]:
        return False
    if succ[h] == n:
        return True
    L = 0
    c = succ[h]
    while c != n:                           # collect the skipped stretch S
        if occ[c]:
            return False
        S[L] = c
        L += 1
        c = succ[c]
    if L % 2 == 1:                          # cannot happen on a grid; be safe
        return False
    nlog = 0
    nlog = _ws(succ, pred, la, li, lo, nlog, 0, h, n)   # the shortcut itself
    nlog = _ws(succ, pred, la, li, lo, nlog, 1, n, h)
    for k in range(L):
        det[S[k]] = 1                       # "skipped, not yet re-added"
    depth = 0
    fi[0] = 0
    fj[0] = L
    fn_[0] = _gen_options(S[0], S[L - 1], n, W, nbrs, succ, occ, det, opts, 0)
    fk[0] = 0
    calls = budget
    ok = False
    while True:
        f = depth
        if fk[f] < fn_[f]:                  # try the next way to glue this piece
            u = opts[f, fk[f], 0]
            v = opts[f, fk[f], 1]
            rev = opts[f, fk[f], 2]
            fk[f] += 1
            calls -= 1
            if calls < 0:
                break                       # out of budget: give up
            fmark[f] = nlog
            i = fi[f]
            j = fj[f]
            a = S[i]
            b = S[j - 1]
            if rev == 0:
                nlog = _ws(succ, pred, la, li, lo, nlog, 0, u, a)
                nlog = _ws(succ, pred, la, li, lo, nlog, 1, a, u)
                nlog = _ws(succ, pred, la, li, lo, nlog, 0, b, v)
                nlog = _ws(succ, pred, la, li, lo, nlog, 1, v, b)
            else:
                for k in range(i + 1, j):
                    nlog = _ws(succ, pred, la, li, lo, nlog, 0, S[k], S[k - 1])
                    nlog = _ws(succ, pred, la, li, lo, nlog, 1, S[k - 1], S[k])
                nlog = _ws(succ, pred, la, li, lo, nlog, 0, u, b)
                nlog = _ws(succ, pred, la, li, lo, nlog, 1, b, u)
                nlog = _ws(succ, pred, la, li, lo, nlog, 0, a, v)
                nlog = _ws(succ, pred, la, li, lo, nlog, 1, v, a)
            for k in range(i, j):
                det[S[k]] = 0
            if j == L:                      # every piece placed: success
                ok = True
                break
            depth += 1                      # now place the remaining pieces
            f = depth
            fi[f] = j
            fj[f] = L
            fn_[f] = _gen_options(S[j], S[L - 1], n, W, nbrs, succ, occ, det, opts, f)
            fk[f] = 0
        else:                               # no more options at this size
            fj[f] -= 2                      # try a shorter first piece
            if fj[f] >= fi[f] + 2:
                fn_[f] = _gen_options(S[fi[f]], S[fj[f] - 1], n, W, nbrs, succ,
                                      occ, det, opts, f)
                fk[f] = 0
            else:                           # dead end: step back and undo
                if depth == 0:
                    break
                depth -= 1
                f = depth
                nlog = _undo(succ, pred, la, li, lo, nlog, fmark[f])
                for k in range(fi[f], fj[f]):
                    det[S[k]] = 1
    if not ok:
        nlog = _undo(succ, pred, la, li, lo, nlog, 0)    # restore the old road
    for k in range(L):
        det[S[k]] = 0
    return ok


# ==========================================================================
# 4. DECIDING ONE MOVE
# ==========================================================================
@njit(cache=True, fastmath=True)
def choose_move(W, N, nbrs, succ, pred, occ, pos, ring, tail_ptr, L, apple,
                cfg, cfgf, work, ctr):
    """Pick the snake's next square.

    Order of thinking:
      1. Ask the GPS for the best route to the apple.
      2. If simply following the ring road is about as short, do that (no need
         to disturb the road).
      3. Otherwise try the GPS's first step as a shortcut; if the road can be
         repaired around it, take it. If not, try a couple of other sensible
         neighbours.
      4. If nothing works, follow the ring road - always safe.
    """
    gcost, stamp, steps, came, hf, hg, hn = (work[0], work[1], work[2],
                                              work[3], work[4], work[5], work[6])
    h = ring[(tail_ptr + L - 1) % N]
    nxt = succ[h]
    frac = L / N
    nascar_on = cfg[C_NASCAR] == 1 and frac >= cfgf[0]
    fear_on = cfg[C_FEAR] == 1 and frac >= cfgf[1]
    # QUICK EXIT (gives the same answer, just sooner): no route can be shorter
    # than the straight-line distance. If the ring road already reaches the
    # apple within that distance, the road IS the best route, so skip the GPS.
    if cfg[C_FOLLOW] >= 0:
        limit = abs(apple % W - h % W) + abs(apple // W - h // W) + cfg[C_FOLLOW]
        c = h
        d = 0
        while c != apple and d <= limit:
            c = succ[c]
            d += 1
        if c == apple and d <= limit:
            return nxt
    ctr[0] += 1
    first, plen = astar_first(h, apple, W, N, nbrs, succ, occ, pos, tail_ptr,
                              nascar_on, fear_on, cfg[C_OFF],
                              cfg[C_NASCAR_COST], cfg[C_FEAR_COST],
                              gcost, stamp, ctr[0], steps, came, hf, hg, hn)
    cand = np.empty(4, dtype=np.int32)
    ncand = 0
    skip_all = False
    if first >= 0:
        if cfg[C_FOLLOW] >= 0:              # cycle distance head -> apple
            d = 0
            c = h
            while c != apple:
                c = succ[c]
                d += 1
            if d <= plen + cfg[C_FOLLOW]:
                skip_all = True             # the road is just as good
        if not skip_all:
            cand[0] = first
            ncand = 1
    if not skip_all:
        # fall-back neighbours: free squares, nearest the apple first
        ax = apple % W
        ay = apple // W
        rest = np.empty(4, dtype=np.int32)
        rkey = np.empty(4, dtype=np.int32)
        nr = 0
        for t in range(4):
            n = nbrs[h, t]
            if n < 0 or occ[n]:
                continue
            if ncand == 1 and n == cand[0]:
                continue
            key = abs(n % W - ax) + abs(n // W - ay)
            q = nr                          # stable insertion sort
            while q > 0 and rkey[q - 1] > key:
                rkey[q] = rkey[q - 1]
                rest[q] = rest[q - 1]
                q -= 1
            rkey[q] = key
            rest[q] = n
            nr += 1
        room = cfg[C_MAXCAND] - ncand
        if room < 0:
            room = 0
        for q in range(min(room, nr)):
            cand[ncand] = rest[q]
            ncand += 1
    for q in range(ncand):
        n = cand[q]
        if n == nxt:
            break
        if repair(h, n, W, N, nbrs, succ, pred, occ, cfg[C_BUDGET], work):
            return n
    return nxt


# ==========================================================================
# 5. PLAYING A WHOLE GAME INSIDE COMPILED CODE
# ==========================================================================
@njit(cache=True)
def _rand(state):
    """Tiny random-number generator (xorshift): same seed -> same game."""
    x = state[0]
    x ^= x << np.uint64(13)
    x ^= x >> np.uint64(7)
    x ^= x << np.uint64(17)
    state[0] = x
    return x


@njit(cache=True)
def _spawn_apple(N, occ, rng, nfree):
    """Drop the apple on a random empty square."""
    if nfree <= 0:
        return -1
    k = int(_rand(rng) % np.uint64(nfree))
    for c in range(N):
        if not occ[c]:
            if k == 0:
                return c
            k -= 1
    return -1


@njit(cache=True, fastmath=True)
def run_game(W, H, seed, cfg, cfgf, max_steps):
    """Play one full game. Returns (moves made, final length, status).
    status: 0 = won (board full), 1 = died (should never happen), 2 = move cap.
    """
    N = W * H
    nbrs = build_nbrs(W, H)
    succ = np.zeros(N, np.int32)
    pred = np.zeros(N, np.int32)
    order = build_cycle(W, H, succ, pred)
    work = make_work(N)
    ctr = np.zeros(1, np.int32)
    occ = np.zeros(N, np.uint8)
    pos = np.zeros(N, np.int32)
    ring = np.zeros(N, np.int32)            # snake body stored as a loop buffer
    for i in range(3):                      # start: 3 squares on the road
        ring[i] = order[i]
        pos[order[i]] = i
        occ[order[i]] = 1
    tail_ptr = 0
    L = 3
    rng = np.zeros(1, np.uint64)
    rng[0] = np.uint64(seed) * np.uint64(2654435761) + np.uint64(88172645463325252)
    for _ in range(4):
        _rand(rng)
    apple = _spawn_apple(N, occ, rng, N - L)
    moves = 0
    status = 2
    while L < N and moves < max_steps:
        nxt = choose_move(W, N, nbrs, succ, pred, occ, pos, ring, tail_ptr, L,
                          apple, cfg, cfgf, work, ctr)
        ate = nxt == apple
        if not ate:                         # tail leaves its square
            occ[ring[tail_ptr]] = 0
            tail_ptr = (tail_ptr + 1) % N
            L -= 1
        if occ[nxt]:
            status = 1
            break
        slot = (tail_ptr + L) % N           # head goes into the next buffer slot
        ring[slot] = nxt
        pos[nxt] = slot
        occ[nxt] = 1
        L += 1
        moves += 1
        if ate:
            apple = _spawn_apple(N, occ, rng, N - L)
    if L == N:
        status = 0
    return moves, L, status


@njit(parallel=True, fastmath=True, cache=True)
def play_batch(W: int, H: int, seeds: Array, cfg: Array, cfgf: Array,
               max_steps: int, out_steps: Array, out_len: Array,
               out_status: Array):
    """Play many independent games at once, one per CPU core."""
    for i in prange(seeds.shape[0]):
        m, L, s = run_game(W, H, seeds[i], cfg, cfgf, max_steps)
        out_steps[i] = m
        out_len[i] = L
        out_status[i] = s


# ==========================================================================
# 6. FRIENDLY PYTHON WRAPPERS
# ==========================================================================
def _cfg(mode):
    return MODES_FAST[mode] if isinstance(mode, str) else mode


def play_many(W, H, seeds, mode="all", max_steps=0):
    """Play one game per seed in parallel. Returns (moves, lengths, statuses)."""
    seeds = np.asarray(seeds, dtype=np.int64)
    ci, cf = _cfg(mode)
    cap = max_steps or (W * H) ** 2 * 20
    o1 = np.zeros(len(seeds), np.int64)
    o2 = np.zeros(len(seeds), np.int64)
    o3 = np.zeros(len(seeds), np.int64)
    play_batch(W, H, seeds, ci, cf, cap, o1, o2, o3)
    return o1, o2, o3


def play_fast(W, H=None, seed=0, mode="all", max_steps=0):
    """Play a single game. Returns a dict with moves / length / status."""
    H = H or W
    ci, cf = _cfg(mode)
    m, L, s = run_game(W, H, seed, ci, cf, max_steps or (W * H) ** 2 * 20)
    return {"moves": int(m), "length": int(L), "won": s == 0, "status": int(s)}


# ==========================================================================
# 7. CHECKS AND BENCHMARK
# ==========================================================================
def check_against_base(size=10, seeds=(0, 1, 2), modes=("plain", "all")):
    """Feed the SAME board situations to the pure-Python solver and to the
    compiled one and count how often they choose a different move."""
    from dhcr import Game, DHCRAgent, MODES
    mism = total = 0
    for mode in modes:
        ci, cf = MODES_FAST[mode]
        for seed in seeds:
            g = Game(size, size, seed=seed)
            agent = DHCRAgent(g, MODES[mode])
            W = N = None
            W, N = g.W, g.N
            nbrs = build_nbrs(W, g.H)
            work = make_work(N)
            ctr = np.zeros(1, np.int32)
            while not g.won():
                succ = np.array(agent.cyc.succ, np.int32)
                pred = np.array(agent.cyc.pred, np.int32)
                ring = np.zeros(N, np.int32)
                body = list(g.snake)
                ring[:len(body)] = body
                pos = np.zeros(N, np.int32)
                pos[body] = np.arange(len(body))
                occ = np.zeros(N, np.uint8)
                occ[body] = 1
                mf = choose_move(W, N, nbrs, succ, pred, occ, pos, ring, 0,
                                 len(body), g.apple, ci, cf, work, ctr)
                mp = agent.choose()
                total += 1
                if mf != mp or not np.array_equal(succ, np.array(agent.cyc.succ)):
                    mism += 1
                g.step(mp)
    return mism, total


def benchmark(sizes=(10, 20), seeds=(0, 1, 2), mode="all", base_sizes=None):
    from dhcr import Game, DHCRAgent, MODES, play
    base_sizes = base_sizes if base_sizes is not None else sizes
    t0 = time.perf_counter()
    play_fast(6, 6, 0, mode)                        # compile (or load cache)
    warm = time.perf_counter() - t0
    print(f"Compile / cache-load time: {warm:.2f}s (one-off)\n")
    print(f"{'size':>6} {'base time':>11} {'base moves':>11} {'fast time':>11} "
          f"{'fast moves':>11} {'speed-up':>9}")
    for s in sizes:
        t0 = time.perf_counter()
        fm = [play_fast(s, s, sd, mode)["moves"] for sd in seeds]
        tf = (time.perf_counter() - t0) / len(seeds)
        if s in base_sizes:
            t0 = time.perf_counter()
            bm = []
            for sd in seeds:
                g = Game(s, s, seed=sd)
                play(g, DHCRAgent(g, MODES[mode]))
                bm.append(g.steps)
            tb = (time.perf_counter() - t0) / len(seeds)
            print(f"{s:>4}x{s:<2}{tb:>10.2f}s{int(np.mean(bm)):>12}{tf:>10.3f}s"
                  f"{int(np.mean(fm)):>12}{tb / tf:>8.0f}x")
        else:
            print(f"{s:>4}x{s:<2}{'-':>11}{'-':>12}{tf:>10.3f}s"
                  f"{int(np.mean(fm)):>12}{'-':>9}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Numba DHCR Snake AI")
    ap.add_argument("--size", type=int, default=30)
    ap.add_argument("--games", type=int, default=1, help="parallel games")
    ap.add_argument("--mode", choices=list(MODES_FAST), default="all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        m, t = check_against_base()
        print(f"decisions differing from pure-Python version: {m}/{t}")
    elif a.bench:
        benchmark()
    else:
        seeds = np.arange(a.seed, a.seed + a.games)
        play_many(4, 4, seeds[:1], a.mode)          # warm-up / compile
        t0 = time.perf_counter()
        mv, ln, st = play_many(a.size, a.size, seeds, a.mode)
        dt = time.perf_counter() - t0
        for s, m, l, k in zip(seeds, mv, ln, st):
            print(f"seed {s}: {'won' if k == 0 else 'FAILED'} "
                  f"{a.size}x{a.size} in {m} moves")
        print(f"{a.games} game(s) in {dt:.3f}s")