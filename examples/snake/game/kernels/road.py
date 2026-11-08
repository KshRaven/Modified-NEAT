"""Road kernels: the Hamiltonian-cycle "ring road" the road reward modes are measured against.

Two roads, both a single one-way cycle through every board square (square = y * W + x):

  dhcr       fixed zigzag cycle. The body is "in road order" when the cycle rank of its cells, counted from the
             tail, grows from tail to head. From such a state the snake can always follow the road without
             hitting itself, and it may hop to a free neighbour that is further along the road but not past
             the food (the tier-1 shortcut of the dhcr solver).
  cell_tree  the cycle is the outline of a spanning tree over 2x2 cells. A move off the road is safe only if a
             tree swap makes it a road link while the body stays one contiguous stretch of the road. The tree
             functions below are the tested ones from the cell-tree solver (`tests/cell_tree.py`).

`scan` grades the three relative moves of every snake (left / straight / right) from that logic and measures the
free room behind each move with a flood fill. The same arrays feed the reward (what the chosen move was) and the
strategy state (what every move would be).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from numpy import ndarray as Array

__all__ = [
    "K_DEAD", "K_ROAD", "K_SHORT", "K_SLOW", "K_UNSAFE", "K_FREE",
    "ROAD_NONE", "ROAD_DHCR", "ROAD_TREE",
    "build_roads", "spawn_road", "scan", "commit", "road_available",
]

# Grade of a relative move
K_DEAD, K_ROAD, K_SHORT, K_SLOW, K_UNSAFE, K_FREE = 0, 1, 2, 3, 4, 5
#   DEAD    wall or body (the tail cell does not count: it is free next frame)
#   ROAD    the next road square
#   SHORT   safe hop that removes more than one cell of road distance to the food
#   SLOW    safe but not shorter (overshooting the food, or a legal tree swap that does not help)
#   UNSAFE  free cell, but taking it breaks road order / has no legal tree swap
#   FREE    free cell while the body is out of road order (no road grade possible)

ROAD_NONE, ROAD_DHCR, ROAD_TREE = 0, 1, 2

_DX = np.array([1, 0, -1, 0], dtype=np.int64)
_DY = np.array([0, -1, 0, 1], dtype=np.int64)

# ==========================================================================
# 1. TREE ROAD (from tests/cell_tree.py)
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


# ==========================================================================
# 2. ROAD CONSTRUCTION
# ==========================================================================
def road_available(road: int, W: int, H: int) -> bool:
    """dhcr needs an even number of squares with one even side, cell_tree two even sides >= 4."""
    if road == ROAD_DHCR:
        return (W * H) % 2 == 0 and (W % 2 == 0 or H % 2 == 0)
    if road == ROAD_TREE:
        return W % 2 == 0 and H % 2 == 0 and W >= 4 and H >= 4
    return True


@njit(cache=True)
def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


@njit(cache=True)
def random_tree(cw, ch, prio, tr, td):
    """Random spanning tree of the cw x ch cells (Kruskal over randomly ordered links)."""
    n = cw * ch
    tr[:] = 0
    td[:] = 0
    parent = np.arange(n)
    order = np.argsort(prio)
    for k in range(order.shape[0]):
        e = order[k]
        c = e >> 1
        if (e & 1) == 1:
            if c // cw == ch - 1:
                continue
            d = c + cw
        else:
            if c % cw == cw - 1:
                continue
            d = c + 1
        ra = _find(parent, c)
        rb = _find(parent, d)
        if ra != rb:
            parent[ra] = rb
            set_edge(e, 1, tr, td)


@njit(cache=True)
def zigzag(W, H, succ, rank):
    """The dhcr starting road: right along the top row, back and forth through the rest, up the free column."""
    N = W * H
    order = np.empty(N, dtype=np.int32)
    k = 0
    if H % 2 == 0:
        w, h = W, H
        swap = False
    else:
        w, h = H, W
        swap = True
    for x in range(w):
        order[k] = (x * W) if swap else x
        k += 1
    for y in range(1, h):
        if y % 2 == 1:
            for x in range(w - 1, 0, -1):
                order[k] = (x * W + y) if swap else (y * W + x)
                k += 1
        else:
            for x in range(1, w):
                order[k] = (x * W + y) if swap else (y * W + x)
                k += 1
    for y in range(h - 1, 0, -1):
        order[k] = (0 * W + y) if swap else (y * W + 0)
        k += 1
    for i in range(N):
        succ[order[i]] = order[(i + 1) % N]
        rank[order[i]] = i


@njit(cache=True)
def rebuild_tree_road(W, cw, N, tr, td, succ, rank, a, b):
    """succ / rank of the tree road, oriented so that succ[a] == b (a and b must be road neighbours)."""
    x, y = cadj(a, W, cw, tr, td)
    prev = y if x == b else x
    cur = a
    for i in range(N):
        rank[cur] = i
        nx = road_next(cur, prev, W, cw, tr, td)
        succ[cur] = nx
        prev = cur
        cur = nx


@njit(parallel=True, cache=True)
def build_roads(mask, road, W, H, random_road, prio, tr, td, succ, rank):
    """(Re)build the road of the masked snakes. `prio` (players, 2 * cells) are random link priorities."""
    N = W * H
    cw = W // 2
    ch = H // 2
    for p in prange(mask.shape[0]):
        if not mask[p]:
            continue
        if road == ROAD_DHCR:
            zigzag(W, H, succ[p], rank[p])
        elif road == ROAD_TREE:
            if random_road:
                random_tree(cw, ch, prio[p], tr[p], td[p])
            else:
                tr[p, :] = 0
                td[p, :] = 0
                init_tree(cw, ch, tr[p], td[p])
            a, b = cadj(0, W, cw, tr[p], td[p])
            rebuild_tree_road(W, cw, N, tr[p], td[p], succ[p], rank[p], 0, b)


@njit(parallel=True, cache=True)
def spawn_road(
    cells, body_x, body_y, head_idx, length, direction, hiatus, action_count,
    mask, start, succ, init_len, W,
):
    """Place the masked snakes on the road: tail on square `start[p]`, body along the road, heading along it."""
    for p in prange(cells.shape[0]):
        if not mask[p]:
            continue
        cells[p, :, :] = 0
        s = start[p]
        for i in range(init_len):                       # i = 0 is the tail
            x = s % W
            y = s // W
            body_x[p, i] = x
            body_y[p, i] = y
            cells[p, x, y] = 2
            s = succ[p, s]
        hx = body_x[p, init_len - 1]
        hy = body_y[p, init_len - 1]
        cells[p, hx, hy] = 3
        dx = s % W - hx
        dy = s // W - hy
        d = 0 if dx == 1 else (2 if dx == -1 else (3 if dy == 1 else 1))
        direction[p] = d
        length[p] = init_len
        head_idx[p] = init_len - 1
        hiatus[p] = 0
        action_count[p, :] = 0


# ==========================================================================
# 3. FREE ROOM (flood fill)
# ==========================================================================
@njit(cache=True)
def flood(start, tail, W, H, wrap, occ, queue, vis, stamp):
    """Squares reachable from `start` over free squares, and whether the tail square borders that region."""
    area = 0
    seen = 0
    if start == tail:
        seen = 1
    qh = 0
    qt = 1
    queue[0] = start
    vis[start] = stamp
    while qh < qt:
        c = queue[qh]
        qh += 1
        area += 1
        x = c % W
        y = c // W
        for d in range(4):
            nx = x + _DX[d]
            ny = y + _DY[d]
            if wrap:
                nx %= W
                ny %= H
            elif nx < 0 or ny < 0 or nx >= W or ny >= H:
                continue
            m = ny * W + nx
            if vis[m] == stamp:
                continue
            if m == tail:
                seen = 1
                vis[m] = stamp
                continue
            if occ[m]:
                continue
            vis[m] = stamp
            queue[qt] = m
            qt += 1
    return area, seen


# ==========================================================================
# 4. SCAN: GRADE THE THREE MOVES OF EVERY SNAKE
# ==========================================================================
@njit(parallel=True, cache=True)
def scan(
    body_x, body_y, head_idx, length, direction, food_x, food_y, active,
    W, H, wrap, road, succ, rank, tr, td, shortcut_frac, any_dir,
    ring, pos, occ, queue, vis, stamp, tq, tpc, tvis, tnb, tpath, tcells, tgen,
    kind, gain, space, tail_ok, trap, cand_rem, cand_add, nbr, phi, synced,
):
    """Fill per snake and per relative move r (0 left, 1 straight, 2 right):

    kind      grade (K_*)                              gain      road cells saved by the move (1 = plain step)
    space     free squares reachable after the move    tail_ok   the region touches the tail
    trap      the move enters a pocket smaller than the snake with no way to the tail
    cand_*    tree swap (link to remove / add) that makes the move legal, -1 when none
    nbr       square of the move (-1 = off the board)
    and per snake: phi = road distance head -> food, synced = body is in road order (always true without a road).
    """
    N = W * H
    cw = W // 2
    chh = H // 2
    for p in prange(body_x.shape[0]):
        for r in range(3):
            kind[p, r] = K_DEAD
            gain[p, r] = 0.0
            space[p, r] = 0
            tail_ok[p, r] = 0
            trap[p, r] = False
            cand_rem[p, r] = -1
            cand_add[p, r] = -1
            nbr[p, r] = -1
        phi[p] = 0.0
        synced[p] = True
        if not active[p]:
            continue
        L = length[p]
        hi = head_idx[p]
        tail_ptr = (hi - (L - 1)) % N
        for s in range(N):
            occ[p, s] = 0
        for i in range(L):
            idx = (tail_ptr + i) % N
            s = body_y[p, idx] * W + body_x[p, idx]
            ring[p, idx] = s
            pos[p, s] = idx
            occ[p, s] = 1
        h = ring[p, hi]
        tail = ring[p, tail_ptr]
        hx = h % W
        hy = h // W
        apple = -1
        if food_x[p] >= 0:
            apple = food_y[p] * W + food_x[p]

        ok = True
        if road == ROAD_DHCR:
            rt = rank[p, tail]
            prev = -1
            for i in range(L):
                rel = (rank[p, ring[p, (tail_ptr + i) % N]] - rt) % N
                if rel <= prev:
                    ok = False
                    break
                prev = rel
        elif road == ROAD_TREE:
            for i in range(L - 1):
                a, b = cadj(ring[p, (tail_ptr + i) % N], W, cw, tr[p], td[p])
                nxt = ring[p, (tail_ptr + i + 1) % N]
                if a != nxt and b != nxt:
                    ok = False
                    break
        use_road = road != ROAD_NONE
        synced[p] = ok or not use_road
        d_cur = 0
        if use_road and ok and apple >= 0:
            d_cur = (rank[p, apple] - rank[p, h]) % N
            phi[p] = d_cur
        ax = apple % W if apple >= 0 else 0
        ay = apple // W if apple >= 0 else 0
        man = abs(hx - ax) + abs(hy - ay)

        for r in range(3):
            d = (direction[p] + (1 if r == 0 else (0 if r == 1 else 3))) % 4
            nx = hx + _DX[d]
            ny = hy + _DY[d]
            if wrap:
                nx %= W
                ny %= H
            elif nx < 0 or ny < 0 or nx >= W or ny >= H:
                continue
            n = ny * W + nx
            nbr[p, r] = n
            if occ[p, n] == 1 and n != tail:
                continue
            area, seen = flood(n, tail, W, H, wrap, occ[p], queue[p], vis[p], stamp[p] + 1)
            stamp[p] += 1
            space[p, r] = area
            tail_ok[p, r] = seen
            trap[p, r] = area < L and seen == 0

            if not use_road or not ok:
                kind[p, r] = K_FREE
                continue
            if n == succ[p, h]:
                kind[p, r] = K_ROAD
                gain[p, r] = 1.0
                continue
            if road == ROAD_DHCR:
                rt = rank[p, tail]
                relh = (rank[p, h] - rt) % N
                reln = (rank[p, n] - rt) % N
                dn = reln - relh
                if dn > 1 and apple >= 0:
                    da = (rank[p, apple] - rt) % N - relh
                    if da >= dn:
                        if L < shortcut_frac * N:
                            kind[p, r] = K_SHORT
                            gain[p, r] = dn
                        else:
                            kind[p, r] = K_SLOW
                            gain[p, r] = 1.0
                    else:                               # hops past the food: a whole lap more
                        kind[p, r] = K_SLOW
                        gain[p, r] = d_cur - (rank[p, apple] - rank[p, n]) % N
                elif dn > 1:
                    kind[p, r] = K_SLOW
                    gain[p, r] = 1.0
                else:
                    kind[p, r] = K_UNSAFE
            else:
                kind[p, r] = K_UNSAFE
                if apple < 0 or n == tail or abs(nx - hx) + abs(ny - hy) != 1:
                    continue
                if d_cur <= man or (any_dir == 0 and abs(nx - ax) + abs(ny - ay) >= man):
                    continue                            # the road is already as short as it gets / not closer
                work = (tq[p], tpc[p], tvis[p], tnb[p], tpath[p], tcells[p], tgen[p])
                bd, rem, add = try_candidate(
                    h, n, W, H, N, cw, chh, tr[p], td[p], occ[p], pos[p], ring[p], tail_ptr, L,
                    apple, N + 1, work,
                )
                if rem >= 0:
                    cand_rem[p, r] = rem
                    cand_add[p, r] = add
                    gain[p, r] = d_cur - bd + 1
                    kind[p, r] = K_SHORT if bd < d_cur else K_SLOW


@njit(parallel=True, cache=True)
def commit(actions, active, road, W, H, kind, cand_rem, cand_add, nbr, ring, head_idx, tr, td, succ, rank):
    """Apply the tree swap behind the chosen move (cell_tree) and re-orient the road along the new head link."""
    if road != ROAD_TREE:
        return
    N = W * H
    cw = W // 2
    for p in prange(actions.shape[0]):
        if not active[p]:
            continue
        r = actions[p]
        if cand_rem[p, r] < 0 or (kind[p, r] != K_SHORT and kind[p, r] != K_SLOW):
            continue
        set_edge(cand_rem[p, r], 0, tr[p], td[p])
        set_edge(cand_add[p, r], 1, tr[p], td[p])
        rebuild_tree_road(W, cw, N, tr[p], td[p], succ[p], rank[p], ring[p, head_idx[p]], nbr[p, r])