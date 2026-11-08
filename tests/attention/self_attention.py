"""
Attention test-suite (uses YOUR real NeatModule / NeatParameter / LayerNorm / Linear / etc.)

Usage (the test can be run from any working directory):
    python tests/attention/self_attention.py [dotted.module.path.of.sub] [--list]
    e.g. python tests/attention/self_attention.py ModifiedNEAT.nn.modules.sub

Every module has genome dim = 1 here, so all tensors are (genomes=1, batch, seq, dim).
All NeatParameters in the Attention under test are re-randomised so the checks are not trivial.
Each check compares against an independent float64 reference (own RoPE / softmax / mask).
"""
import sys, math, importlib, traceback, torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

mod_path = next((a for a in sys.argv[1:] if not a.startswith('--')), 'ModifiedNEAT.nn.modules.sub')
sub = importlib.import_module(mod_path)
# Attention, NeatParameter = sub.Attention, sub.NeatParameter

from ModifiedNEAT.nn import Genome, Attention, NeatParameter, NeatModule

LIST_ONLY = '--list' in sys.argv
path = mod_path

def neat_params(m):
    """All NeatParameters under module m (scans attributes and neat_parameters()), de-duplicated."""
    found = {}
    for sm in m.modules():
        for v in vars(sm).values():
            if isinstance(v, NeatParameter): found[id(v)] = v
        if hasattr(sm, 'neat_parameters'):
            try:
                for v in sm.neat_parameters(): found[id(v)] = v
            except Exception: pass
    return list(found.values())

def _fill(p, std):
    """Re-randomise p. Try in-place first; if that is illegal or doesn't stick (e.g. the genome rows are an
    expand()ed view, so several rows alias one memory location), assign a fresh contiguous tensor instead."""
    before = p.data.clone()
    try:
        with torch.no_grad(): p.data.normal_(0, std)
        if not torch.equal(before, p.data): return 'in-place'
    except RuntimeError:
        pass                      # "more than one element of the written-to tensor refers to a single memory location"
    fresh = torch.randn(before.shape, dtype=before.dtype, device=before.device) * std
    try:
        with torch.no_grad(): p.data = fresh
        if not torch.equal(before, p.data): return 'assigned'
    except Exception: pass
    return None

RANDOMIZE_STATS = {'in-place': 0, 'assigned': 0, 'FAILED': 0}
def randomize(m, std=0.3):
    """Fill every NeatParameter in the module tree with N(0, std) so nothing is zero/identity."""
    for p in neat_params(m):
        RANDOMIZE_STATS[_fill(p, std) or 'FAILED'] += 1
    return m

# ======================= test definitions =======================
torch.manual_seed(1)
DT = torch.float64
D, H, MAX = 16, 4, 10
CHECKS = []                      # (name, fn)
def check(name, f): CHECKS.append((name, f))

def mk(kv=None, diff=None, layer=0, mx=MAX, **kw):
    return randomize(Attention(mx, D, H, kv, diff, layer, device='cpu', dtype=DT, **kw))

# ---------- independent reference (own RoPE, own softmax attention) ----------
def rope_ref(x, pos0):
    s, d = x.shape[2], x.shape[-1]
    theta = 1.0 / torch.pow(10000, torch.arange(0, d, 2).float() / d)          # same f32 precision as the module
    ang = torch.outer(pos0 + torch.arange(s), theta).to(x.dtype)
    c, sn = ang.cos()[None, None, :, None, :], ang.sin()[None, None, :, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1*c - x2*sn, x1*sn + x2*c], -1).flatten(-2)

def rope_branch_ref(x, pos0, mult, hd, branch):
    """branch=True: every differential branch is rotated as its own head_dim-wide vector (same frequencies)."""
    if branch and mult > 1:
        shp = x.shape
        return rope_ref(x.reshape(*shp[:-2], shp[-2] * mult, hd), pos0).reshape(shp)
    return rope_ref(x, pos0)

def qkn_ref(x, mult, hd, eps):
    xr = x.reshape(*x.shape[:-1], mult, hd)
    return (xr / torch.sqrt(xr.pow(2).mean(-1, keepdim=True) + eps)).flatten(-2)

def ref(m, x, keys=None, prompt=None, q_start=0, k_start=0):
    resid = x if prompt is None else prompt
    xn = m.pre_norm(x, keys=keys); pn = xn if prompt is None else m.pre_norm(prompt, keys=keys)
    g, b, K, _ = x.shape; Q = pn.shape[2]; mult = m.mult; hd = m.head_dim
    q = m.query_proj(pn, keys=keys).view(g, b, Q, m.heads, hd*mult)
    k = m.key_proj(xn, keys=keys).view(g, b, K, m.kv_heads, hd*mult)
    v = m.value_proj(xn, keys=keys).view(g, b, K, m.kv_heads, hd)
    if getattr(m, 'qk_norm', False): q, k = qkn_ref(q, mult, hd, m.epsilon), qkn_ref(k, mult, hd, m.epsilon)
    br = getattr(m, 'branch_rope', False)
    q, k = rope_branch_ref(q, q_start, mult, hd, br), rope_branch_ref(k, k_start, mult, hd, br)
    r = m.heads // m.kv_heads
    k, v = k.repeat_interleave(r, dim=3), v.repeat_interleave(r, dim=3)
    qc, kc = q.view(g, b, Q, m.heads, mult, hd), k.view(g, b, K, m.heads, mult, hd)
    e = torch.einsum('gbqhcd,gbkhcd->gbhcqk', qc, kc)                  # (g,b,h,c,q,k)
    if m.differential:
        lam = m.diff_lambda(keys)                                      # (g,1,h,l,1,1)
        e = e[:, :, :, 0] + (e[:, :, :, 1:] * lam).sum(3)
    else:
        e = e[:, :, :, 0]
    mask = torch.arange(K)[None, :] > (torch.arange(Q)[:, None] + K - Q)
    e = e.masked_fill(mask, -torch.inf)
    p = torch.softmax(e / math.sqrt(hd), -1)
    o = torch.einsum('gbhqk,gbkhd->gbqhd', p, v)
    o = m.head_norm(o, keys=keys).reshape(g, b, Q, m.dim_size)
    return m.out_proj(o, keys=keys) + resid

def close(a, b, tol=1e-9):
    err = (a - b).abs().max().item(); return err < tol, f"max|err|={err:.1e}"

X = lambda s, g=1, b=1: torch.randn(g, b, s, D, dtype=DT)

# ---------- A. plain attention ----------
for name, kw in [("MHA", {}), ("GQA kv=2", dict(kv=2)), ("MQA kv=1", dict(kv=1)),
                 ("diff=1", dict(diff=1)), ("diff=2 + GQA", dict(diff=2, kv=2))]:
    m = mk(**kw); x = X(MAX, b=2)
    check(f"A {name}: self-attn q=k=10", lambda m=m, x=x: close(m(x), ref(m, x)))
    m14 = mk(mx=14, **kw); x12, p10 = X(12), X(10)
    check(f"A {name}: q=10, k=12 (prompt), pos_idx=2", lambda m14=m14, x12=x12, p10=p10: close(m14(x12, prompt=p10, pos_idx=2), ref(m14, x12, prompt=p10, q_start=2, k_start=0)))
    p1 = X(1)
    check(f"A {name}: q=1,  k=10 (prompt), pos_idx=-1", lambda m=m, x=x: close(
        m(x, prompt=x[:, :, -1:], pos_idx=-1), ref(m, x, prompt=x[:, :, -1:], q_start=MAX-1, k_start=0)))

# ---------- B. normal cache ----------
for name, kw in [("MHA", {}), ("GQA+diff", dict(kv=2, diff=1))]:
    m = mk(**kw); x = X(MAX)
    full = m(x)
    def step(m=m, x=x, full=full):
        m.empty_cache(); outs = [m(x[:, :, t:t+1], pos_idx=t) for t in range(MAX)]
        return close(torch.cat(outs, 2), full)
    check(f"B {name}: normal cache, s=1 steps == full run", step)
    def chunks(m=m, x=x, full=full):
        m.empty_cache(); o = [m(x[:, :, :6], pos_idx=0), m(x[:, :, 6:], pos_idx=6)]
        return close(torch.cat(o, 2), full)
    check(f"B {name}: normal cache, chunks 6+4 == full run", chunks)
    def rewind(m=m, x=x):
        m.empty_cache(); m(x, pos_idx=0)
        new = X(1); o = m(new, pos_idx=5)
        want = m(torch.cat([x[:, :, :5], new], 2))[:, :, -1:]
        return close(o, want)
    check(f"B {name}: rewind to pos 5 truncates cache", rewind)

# ---------- C. forced cache with dequeue (layer 0 => repositioning is exact) ----------
def forced(kw, sizes):
    m = mk(**kw); m.update({0: Genome(0, 0)}); m.force_cache(True); m.empty_cache()
    toks = X(sum(sizes), b=1); i = 0; worst = 0.0
    for s in sizes:
        chunk = toks[:, :, i:i+s]; i += s
        out = m(chunk)
        win = toks[:, :, max(0, i-MAX):i]
        # reference = non-cached run over the retained window (positions 0..W-1), last s rows
        want = ref(m, win)[:, :, -s:]
        worst = max(worst, (out - want).abs().max().item())
        exp_size = min(i, MAX)
        if m.cache_size != exp_size: return False, f"cache_size {m.cache_size} != {exp_size} after {i} tokens"
    return worst < 1e-9, f"max|err|={worst:.1e}"
for name, kw in [("MHA", {}), ("GQA", dict(kv=2)), ("diff=1", dict(diff=1)), ("GQA+diff=2", dict(kv=2, diff=2))]:
    check(f"C {name}: forced, 1 token x 25", lambda kw=kw: forced(kw, [1]*25))
    check(f"C {name}: forced, s=max then 1s", lambda kw=kw: forced(kw, [MAX] + [1]*5))
    check(f"C {name}: forced, s=max-1 repeated", lambda kw=kw: forced(kw, [MAX-1]*4))
    check(f"C {name}: forced, s=max repeated (full replace)", lambda kw=kw: forced(kw, [MAX]*3))
    check(f"C {name}: forced, mixed 3,1,4,10,2,9", lambda kw=kw: forced(kw, [3,1,4,10,2,9]))

# ---------- D. misc ----------
def l1_plain(keys):
    def f():
        m = mk(layer=1); m.update({0: Genome(0, 0)})
        o = m(X(3), keys=keys)
        return o.shape == (1, 1, 3, D), "plain forward (no cache)"
    return f
check("D layer>0, keys=None, plain forward", l1_plain(None))
check("D layer>0, keys=[0],  plain forward", l1_plain([0]))
def l1_forced(keys):
    def f():
        m = mk(layer=1); m.update({0: Genome(0, 0)}); m.force_cache(True)
        for _ in range(MAX + 3): o = m(X(1), keys=keys)
        return o.shape == (1, 1, 1, D) and m.cache_size == MAX, f"cache_size={m.cache_size}"
    return f
check("D layer>0, forced cache, keys=None, overflow (learned repositioning)", l1_forced(None))
check("D layer>0, forced cache, keys=[0],  overflow (learned repositioning)", l1_forced([0]))
# ---------- E. multiple genomes: batched keys=[0,1] must equal each genome run alone with keys=[g] ----------
def randomize_works():
    m = mk(layer=1); m.update({0: Genome(0, 0), 1: Genome(1, 0)}); randomize(m)
    ps = neat_params(m); w = m.query_proj.weights
    rows = (w.data[0] - w.data[1]).abs().max().item()
    return (RANDOMIZE_STATS['FAILED'] == 0 and rows > 1e-6 and len(ps) > 0,
            f"{len(ps)} NeatParameters found, fill methods so far={RANDOMIZE_STATS}, query_proj row0-vs-row1 diff={rows:.1e}")
check("E0 randomize() really changes every NeatParameter (incl. the 2nd genome's row)", randomize_works)

def diag(m):
    """Why are two genomes identical? Show the parameter storage behind keys 0 and 1."""
    try:
        w = m.query_proj.weights
        d = w.data
        rows = f"row0-vs-row1 diff={(d[0] - d[1]).abs().max().item():.1e}" if d.shape[0] > 1 else "only 1 row"
        return f"query_proj.weights.data.shape={tuple(d.shape)}, {rows}, mapping={getattr(w, 'mapping', 'n/a')}"
    except Exception as e:
        return f"diag failed: {type(e).__name__}: {e}"

def multi(kw, layer):
    def f():
        m = mk(layer=layer, **kw); m.update({0: Genome(0, 0), 1: Genome(1, 0)}); randomize(m); m.force_cache(True)
        N = MAX + 5                                              # past overflow, so repositioning runs
        T = torch.randn(1, 1, N, D, dtype=DT).expand(2, 1, N, D).clone()   # same tokens for both genomes
        m.empty_cache()
        both = torch.cat([m(T[:, :, t:t+1], keys=[0, 1]) for t in range(N)], 2)
        worst = 0.0
        for g in (0, 1):
            m.empty_cache()
            solo = torch.cat([m(T[g:g+1, :, t:t+1], keys=[g]) for t in range(N)], 2)
            worst = max(worst, (both[g:g+1] - solo).abs().max().item())
        differ = (both[0] - both[1]).abs().max().item()          # genomes must actually behave differently
        ok = worst < 1e-9 and differ > 1e-6
        return ok, f"max|err|={worst:.1e}, genome diff={differ:.1e}" + ("" if ok else "\n        " + diag(m))
    return f
check("E 2 genomes, layer 0 (exact RoPE repositioning): batched == solo", multi({}, 0))
check("E 2 genomes, layer>0 (learned repositioning): batched == solo", multi({}, 1))
check("E 2 genomes, layer>0, GQA+diff=1: batched == solo", multi(dict(kv=2, diff=1), 1))

# ---------- E2. each genome gets its OWN tokens (and batch>1): no cross-genome leakage ----------
def multi_distinct(kw, layer, sizes, B=2):
    def f():
        m = mk(layer=layer, **kw); m.update({0: Genome(0, 0), 1: Genome(1, 0)}); randomize(m); m.force_cache(True)
        N = sum(sizes)
        T = torch.randn(2, B, N, D, dtype=DT)                      # different tokens per genome AND per batch row
        def batched(T):
            m.empty_cache(); i = 0; outs = []
            for s in sizes: outs.append(m(T[:, :, i:i+s], keys=[0, 1])); i += s
            return torch.cat(outs, 2)
        def solo(T, g):
            m.empty_cache(); i = 0; outs = []
            for s in sizes: outs.append(m(T[g:g+1, :, i:i+s], keys=[g])); i += s
            return torch.cat(outs, 2)
        both = batched(T)
        worst = max((both[g:g+1] - solo(T, g)).abs().max().item() for g in (0, 1))
        T2 = T.clone(); T2[0] = torch.randn(B, N, D, dtype=DT)      # change ONLY genome 0's inputs
        both2 = batched(T2)
        leak = (both2[1] - both[1]).abs().max().item()              # genome 1 must not notice
        moved = (both2[0] - both[0]).abs().max().item()             # genome 0 must notice
        ok = worst < 1e-9 and leak < 1e-9 and moved > 1e-6
        return ok, f"batched-vs-solo={worst:.1e}, leak into genome 1={leak:.1e}, genome 0 moved={moved:.1e}"
    return f
for lay in (0, 1):
    check(f"E2 layer={lay}, distinct tokens, batch=2, 1 token x 15 (overflow)", multi_distinct({}, lay, [1]*15))
    check(f"E2 layer={lay}, distinct tokens, batch=2, mixed 3,1,4,10,2,9", multi_distinct({}, lay, [3,1,4,10,2,9]))
    check(f"E2 layer={lay}, distinct tokens, batch=2, GQA+diff=1, s=max-1 x4", multi_distinct(dict(kv=2, diff=1), lay, [MAX-1]*4))

# ---------- F. differential gain design ----------
def f_structure(diff, std):
    """simplex coefficients: formula-exact, all <= 0, sum = -beta in [-1, 0] for ANY parameter magnitude."""
    def f():
        m = mk(diff=diff); dl = m.diff_lambda
        _fill(dl.scalars, std)
        lam = dl(None)
        if lam.shape != (1, 1, H, diff, 1, 1): return False, f"shape {tuple(lam.shape)}"
        c = lam[:, 0, :, :, 0, 0]                                                  # (g, h, L)
        s = dl.scalars.data
        want = -torch.sigmoid(s[..., diff:] + dl.beta_bias) * torch.softmax(s[..., :diff], -1)
        err = (c - want).abs().max().item(); tot = c.sum(-1)
        ok = (torch.isfinite(c).all().item() and err < 1e-12 and (c <= 0).all().item() and (c >= -1).all().item()
              and (tot <= 1e-12).all().item() and (tot >= -1 - 1e-12).all().item())
        return ok, f"formula err={err:.1e}, sum(lambda) in [{tot.min().item():.3f}, {tot.max().item():.3f}]"
    return f
for diff in (1, 2, 3):
    for std in (0.3, 5.0, 50.0):
        check(f"F diff={diff}: simplex coefficients, scalar std={std} (sum in [-1,0], formula-exact)", f_structure(diff, std))

def identical_branches(m):
    """Make every differential branch of q and k identical => e_0 = e_1 = ... (pure common-mode signal)."""
    hd, mult = m.head_dim, m.mult
    with torch.no_grad():
        for proj, nh in ((m.query_proj, m.heads), (m.key_proj, m.kv_heads)):
            w = proj.weights.data
            v = w.view(w.shape[0], w.shape[1], nh, mult, hd)
            v[:, :, :, 1:] = v[:, :, :, :1]

def common_mode(diff, mode, kv=None, expect_reject=None, **extra):
    def f():
        m = mk(diff=diff, kv=kv, lambda_mode=mode, **extra); identical_branches(m)
        if mode == 'simplex':
            with torch.no_grad(): m.diff_lambda.scalars.data[..., -1] = 40.0       # beta -> 1
        _, p = m(X(MAX, b=2), get=True)                                            # (g, b, h, q, k)
        want = torch.tril(torch.ones(MAX, MAX, dtype=DT)); want = want / want.sum(-1, keepdim=True)
        dev = (p - want).abs().max().item()
        if expect_reject: return dev < 1e-9, f"deviation from uniform={dev:.1e}"
        return dev > 1e-6, f"leaks the common-mode signal (expected): deviation from uniform={dev:.1e}"
    return f
for diff in (1, 2, 3):
    check(f"F diff={diff}: simplex rejects a pure common-mode signal (attention is uniform)", common_mode(diff, 'simplex', expect_reject=True))
check("F diff=2 + GQA: simplex rejects a pure common-mode signal", common_mode(2, 'simplex', kv=2, expect_reject=True))
check("F diff=2: legacy gains do NOT reject it (documents why simplex exists)", common_mode(2, 'legacy', expect_reject=False))
check("F diff=2: simplex but branch_rope=False does NOT fully reject it (positional part leaks)", common_mode(2, 'simplex', branch_rope=False, expect_reject=False))

def f_beta_zero():
    """beta -> 0 switches the subtraction off: every lambda -> 0, i.e. plain attention on branch 0."""
    m = mk(diff=2)
    with torch.no_grad(): m.diff_lambda.scalars.data[..., -1] = -60.0
    lam = m.diff_lambda(None)
    return lam.abs().max().item() < 1e-12, f"max|lambda|={lam.abs().max().item():.1e}"
check("F diff=2: beta -> 0 gives lambda = 0 (plain attention)", f_beta_zero)

def f_multi_genome():
    m = mk(diff=2, kv=2); m.update({0: Genome(0, 0), 1: Genome(1, 0)}); randomize(m)
    both = m.diff_lambda([0, 1]); w0 = m.diff_lambda([0]); w1 = m.diff_lambda([1])
    err = max((both[0:1] - w0).abs().max().item(), (both[1:2] - w1).abs().max().item())
    differ = (both[0] - both[1]).abs().max().item()
    return both.shape == (2, 1, H, 2, 1, 1) and err < 1e-12 and differ > 1e-4, f"batched-vs-solo={err:.1e}, genome diff={differ:.1e}"
check("F diff=2: per-genome coefficients, keys=[0,1] == solo", f_multi_genome)

# agreement with the independent reference in every mode / with qk_norm
for name, kw in [("simplex diff=1", dict(diff=1)), ("simplex diff=3 + GQA", dict(diff=3, kv=2)),
                 ("legacy diff=2", dict(diff=2, lambda_mode='legacy')),
                 ("branch_rope=False diff=2", dict(diff=2, branch_rope=False)), ("qk_norm MHA", dict(qk_norm=True)), ("qk_norm GQA+diff=2", dict(diff=2, kv=2, qk_norm=True))]:
    m = mk(**kw); x = X(MAX, b=2); m14 = mk(mx=14, **kw); x12, p10 = X(12), X(10)
    check(f"F {name}: self-attn q=k=10 vs reference", lambda m=m, x=x: close(m(x), ref(m, x)))
    check(f"F {name}: q=10, k=12 (prompt), pos_idx=2 vs reference", lambda m14=m14, x12=x12, p10=p10: close(
        m14(x12, prompt=p10, pos_idx=2), ref(m14, x12, prompt=p10, q_start=2, k_start=0)))
    check(f"F {name}: forced, 1 token x 25", lambda kw=kw: forced(kw, [1]*25))
    check(f"F {name}: forced, mixed 3,1,4,10,2,9", lambda kw=kw: forced(kw, [3,1,4,10,2,9]))

def f_qk_scale():
    """With qk_norm the output must not depend on the scale of the q/k projection weights."""
    m = mk(diff=2, qk_norm=True); x = X(MAX, b=2)
    base = m(x)
    with torch.no_grad():
        m.query_proj.weights.data *= 100.0; m.key_proj.weights.data *= 100.0
    return close(m(x), base, 1e-6)
check("F qk_norm: output is invariant to q/k weight scale (bounded energies)", f_qk_scale)

def f_qk_saturated():
    """Huge q/k weights without qk_norm vs with: with it the attention stays finite and non-degenerate."""
    m = mk(diff=2, qk_norm=True); x = X(MAX)
    with torch.no_grad():
        m.query_proj.weights.data *= 1e4; m.key_proj.weights.data *= 1e4
    _, p = m(x, get=True)
    ent = -(p.clamp_min(1e-30) * p.clamp_min(1e-30).log()).sum(-1)[..., -1].mean().item()   # entropy of last row
    return bool(torch.isfinite(p).all()) and ent > 0.05, f"finite, last-row entropy={ent:.2f}"
check("F qk_norm: saturated q/k weights keep attention finite and non-degenerate", f_qk_saturated)

def f_multi_distinct_diff():
    return multi_distinct(dict(diff=3, kv=2, qk_norm=True), 1, [MAX-1]*3)()
check("F simplex diff=3 + GQA + qk_norm, 2 genomes, distinct tokens, no leakage", f_multi_distinct_diff)

# ---------- G. polynomial projections (proj_coeffs > 1 replaces Linear with Polynomial) ----------
import copy
PROJ = ('query_proj', 'key_proj', 'value_proj', 'out_proj')

def g_types(kw, want):
    def f():
        m = mk(**kw); got = tuple(type(getattr(m, n)).__name__ for n in PROJ)
        shapes = [tuple(getattr(m, n).weights.data.shape) for n in PROJ]
        return got == want, f"{got}, weight shapes {shapes}"
    return f
P, L = 'Polynomial', 'Linear'
check("G proj_coeffs=1 keeps Linear everywhere", g_types(dict(proj_coeffs=1), (L, L, L, L)))
check("G proj_coeffs=3 -> Polynomial on q/k/v/out", g_types(dict(proj_coeffs=3), (P, P, P, P)))
check("G proj_coeffs=2, poly_projs='qk' -> only q/k are Polynomial", g_types(dict(proj_coeffs=2, poly_projs='qk'), (P, P, L, L)))

def g_poly1_is_linear():
    lin = sub.Linear(D, 8, True, 'cpu', DT); pol = sub.Polynomial(D, 8, 1, True, 'cpu', DT)
    randomize(lin); randomize(pol)
    with torch.no_grad():
        pol.weights.data = lin.weights.data.unsqueeze(1).clone(); pol.biases.data = lin.biases.data.clone()
    return close(pol(X(5)), lin(X(5).fill_(0) + 1 if False else X(5)))
def g_poly1_is_linear():
    lin = sub.Linear(D, 8, True, 'cpu', DT); pol = sub.Polynomial(D, 8, 1, True, 'cpu', DT)
    randomize(lin); randomize(pol)
    with torch.no_grad():
        pol.weights.data = lin.weights.data.unsqueeze(1).clone(); pol.biases.data = lin.biases.data.clone()
    x = X(5)
    return close(pol(x), lin(x))
check("G Polynomial(coefficients=1) == Linear (same weights)", g_poly1_is_linear)

def g_zero_higher_terms(C):
    """Zeroing every coefficient except the x^1 one must reproduce the same Attention built with Linear layers;
    and with the higher terms present the output must actually differ (they matter)."""
    def f():
        m = mk(proj_coeffs=C, diff=1); x = X(MAX, b=2)
        full = m(x); m2 = copy.deepcopy(m)
        for n in PROJ:
            w = getattr(m2, n).weights.data                                  # (g, C, in, out); last = x^1 coefficient
            lin = sub.Linear(w.shape[2], w.shape[3], False, 'cpu', DT)
            with torch.no_grad(): lin.weights.data = w[:, -1].clone()
            setattr(m2, n, lin)
        with torch.no_grad():
            for n in PROJ: getattr(m, n).weights.data[:, :-1] = 0
        ok, info = close(m(x), m2(x))
        matter = (full - m2(x)).abs().max().item()
        return ok and matter > 1e-6, f"{info}, effect of the higher-order terms={matter:.1e}"
    return f
for C in (2, 3):
    check(f"G proj_coeffs={C}: only x^1 coefficient left == Linear attention; higher terms matter", g_zero_higher_terms(C))

for name, kw in [("C=2", dict(proj_coeffs=2)), ("C=3 GQA+diff=2", dict(proj_coeffs=3, kv=2, diff=2)),
                 ("C=2 bias=True", dict(proj_coeffs=2, bias=True)), ("C=2 q/k only", dict(proj_coeffs=2, poly_projs='qk')),
                 ("C=2 diff=1 + qk_norm", dict(proj_coeffs=2, diff=1, qk_norm=True))]:
    m = mk(**kw); x = X(MAX, b=2); m14 = mk(mx=14, **kw); x12, p10 = X(12), X(10)
    check(f"G {name}: self-attn q=k=10 vs reference", lambda m=m, x=x: close(m(x), ref(m, x)))
    check(f"G {name}: q=10, k=12 (prompt), pos_idx=2 vs reference", lambda m14=m14, x12=x12, p10=p10: close(
        m14(x12, prompt=p10, pos_idx=2), ref(m14, x12, prompt=p10, q_start=2, k_start=0)))
    check(f"G {name}: forced, 1 token x 25", lambda kw=kw: forced(kw, [1]*25))
    check(f"G {name}: forced, mixed 3,1,4,10,2,9", lambda kw=kw: forced(kw, [3,1,4,10,2,9]))
for lay in (0, 1):
    check(f"G layer={lay}: C=3 GQA+diff=2, 2 genomes, distinct tokens, no leakage",
          multi_distinct(dict(proj_coeffs=3, diff=2, kv=2), lay, [3,1,4,10,2,9]))

def g_layers():
    t = sub.TransformerBase(MAX, D, 3, H, None, 1, True, False, 'cpu', DT, proj_coeffs=2, lambda_mode='legacy', qk_norm=True)
    a = [l.self_attention for l in t.layers]
    ok = all(x.lambda_mode == 'legacy' and x.qk_norm and x.proj_coeffs == 2 and type(x.query_proj).__name__ == 'Polynomial' for x in a)
    return ok, f"{len(a)} layers all received proj_coeffs / lambda_mode / qk_norm"
check("G TransformerBase passes the options to every layer", g_layers)

# ---------- H. polynomial Repositioning converters (repo_coeffs > 1; learned path => layer > 0) ----------
def h_types(kw, want):
    def f():
        m = mk(layer=1, **kw)
        got = tuple(type(r.converter).__name__ for r in (m.repo_k, m.repo_v))
        return got == want, f"repo_k/repo_v converters: {got}"
    return f
check("H repo_coeffs=1 keeps Linear converters", h_types(dict(repo_coeffs=1), (L, L)))
check("H repo_coeffs=3 -> Polynomial converters", h_types(dict(repo_coeffs=3), (P, P)))
check("H repo_coeffs=2 is independent of proj_coeffs (q/k/v/o stay Linear)",
      lambda: (lambda m: (all(type(getattr(m, n)).__name__ == 'Linear' for n in PROJ), "projections still Linear"))(mk(layer=1, repo_coeffs=2)))

def h_zero_higher(C, kw):
    """Keep only the x^1 coefficient in both converters => identical to Linear converters; with the higher terms the
    outputs must differ after the cache overflows (so the polynomial path really runs)."""
    def f():
        m = mk(layer=1, repo_coeffs=C, **kw); m.update({0: Genome(0, 0)}); m.force_cache(True)
        toks = X(MAX + 6)
        def run(mod):
            mod.empty_cache(); return torch.cat([mod(toks[:, :, t:t+1]) for t in range(toks.shape[2])], 2)
        full = run(m); m2 = copy.deepcopy(m)
        for r in (m2.repo_k, m2.repo_v):
            c = r.converter; w = c.weights.data                       # (g, C, in, out); last = x^1 coefficient
            lin = sub.Linear(w.shape[2], w.shape[3], c.biases is not None, 'cpu', DT)
            with torch.no_grad():
                lin.weights.data = w[:, -1].clone()
                if c.biases is not None: lin.biases.data = c.biases.data.clone()
            r.converter = lin
        with torch.no_grad():
            for r in (m.repo_k, m.repo_v): r.converter.weights.data[:, :-1] = 0
        a, b = run(m), run(m2)
        ok, info = close(a, b)
        matter = (full - b).abs().max().item()
        return ok and matter > 1e-6 and m.cache_size == MAX, f"{info}, effect of higher-order terms={matter:.1e}, cache_size={m.cache_size}"
    return f
for C in (2, 3):
    check(f"H repo_coeffs={C}: x^1-only == Linear converter; higher terms matter after overflow", h_zero_higher(C, {}))
check("H repo_coeffs=2, bias=True, GQA+diff=2: x^1-only == Linear converter", h_zero_higher(2, dict(bias=True, kv=2, diff=2)))

for lay in (0, 1):
    check(f"H layer={lay}: repo_coeffs=3 + proj_coeffs=2, GQA+diff=2, 2 genomes, distinct tokens, no leakage",
          multi_distinct(dict(repo_coeffs=3, proj_coeffs=2, diff=2, kv=2), lay, [1]*15))
    check(f"H layer={lay}: repo_coeffs=3, mixed 3,1,4,10,2,9, 2 genomes, no leakage",
          multi_distinct(dict(repo_coeffs=3), lay, [3,1,4,10,2,9]))
def h_layer0_exact():
    """Layer 0 uses exact RoPE repositioning: the polynomial converter must not change anything there."""
    return forced(dict(repo_coeffs=3, diff=2, kv=2), [3,1,4,10,2,9])
check("H layer 0 (exact RoPE repositioning) is unaffected by repo_coeffs=3", h_layer0_exact)

def no_cache_flag():
    m = mk(cache=False)
    try: m.force_cache(True)
    except ValueError: return True, "clear error"
    for _ in range(MAX+1): m(X(1))
    return True, ""
check("D cache=False + force_cache", no_cache_flag)
def qgk():
    m = mk()
    try: m(X(4), prompt=X(6)); return False, "no error"
    except ValueError: return True, "clear error"
check("D q_len > k_len rejected", qgk)


# ======================= runner =======================
if LIST_ONLY:
    for i, (n, _) in enumerate(CHECKS, 1): print(f"{i:2d}. {n}")
    sys.exit(0)
print(f"Testing: {path}   ({len(CHECKS)} checks)\n")
passed = 0
for i, (name, f) in enumerate(CHECKS, 1):
    where = ''
    try: ok, info = f()
    except Exception as e:
        ok, info = False, f"{type(e).__name__}: {str(e)[:70]}"
        fr = traceback.extract_tb(e.__traceback__)[-4:]
        where = ''.join(f"\n        at {Path(x.filename).name}:{x.lineno} in {x.name}()  ->  {x.line}" for x in fr)
    passed += ok
    print(f"{i:2d}. [{'PASS' if ok else 'FAIL'}] {name}  {info}{where}")
print(f"\n{passed}/{len(CHECKS)} passed")
sys.exit(0 if passed == len(CHECKS) else 1)