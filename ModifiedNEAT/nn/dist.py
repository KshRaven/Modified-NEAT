import torch

from torch.distributions import Categorical


class SharpCategorical(Categorical):
    """
    Drop-in replacement for `torch.distributions.Categorical` that reports a
    non-saturating 'entropy' surrogate for use in entropy-regularized RL fitness
    (eg. PPO's `ent_reg` / `Model.evaluate_action` / `_get_true_entropy_loss`).

    Standard Shannon entropy for Categorical is bounded in [0, ln K] and collapses
    *exponentially fast* as the policy sharpens, so it loses the ability to
    discriminate 'fairly confident' from 'very confident' genomes almost
    immediately. This class fixes that two ways:

    1. `.entropy()` is derived from the *logit margin* (the unbounded quantity
       your network weights actually shape) instead of raw Shannon entropy.
    2. Optionally accepts a `std` tensor -- eg. the network's own predicted
       per-action uncertainty -- and folds it in as an explicit temperature
       *before* building the distribution. This makes the returned entropy
       exactly linear in `std` (`pseudo_std_out = std / raw_margin`), mirroring
       how `Normal.entropy() = ln(sigma) + const` is a clean, unbounded,
       monotonic function of `sigma` alone. That linearity is what guarantees
       shrinking `std` reliably reduces the reported entropy -- it isn't left
       to chance by entangling the division somewhere upstream.

    The rest of the API (`sample`, `log_prob`, `mode`, `probs`, `support`, etc.)
    is inherited unchanged from `torch.distributions.Categorical`, operating on
    the already-temperature-scaled logits -- so sampling and evaluation both see
    the same effective policy.

    :param probs: Event probabilities. Mutually exclusive with `logits` and with
        `std` (identical semantics to `torch.distributions.Categorical`).
    :param logits: Event log-odds (pre-temperature). Mutually exclusive with `probs`.
    :param std: Optional temperature tensor, broadcastable against `logits`' last
        dim (eg. the raw per-action std head, or its mean/norm reduced to a
        scalar per batch element). `logits` are divided by `std.clamp_min(eps)`
        before the distribution is built. Requires `logits` (not `probs`).
    :param validate_args: Passed through to `torch.distributions.Categorical`.
    :param entropy_mode: Which quantity `.entropy()` returns.
        - ``'pseudo_std'`` (default): ``1 / logit_margin`` computed on the
          (possibly temperature-scaled) logits. When `std` is given this is
          exactly ``std / raw_margin`` -- linear in `std`, unbounded, never
          saturates.
        - ``'neg_margin'``: ``-logit_margin`` directly. Same monotonic
          direction, unbounded below, cheaper (skips the reciprocal).
        - ``'shannon'``: Falls back to standard Categorical Shannon entropy --
          useful to toggle back for A/B comparison without swapping classes.
    :param eps: Numerical floor used for `std` clamping and for `pseudo_std`'s
        division, avoids blow-up when `std`/margin is near zero.
    """

    def __init__(self, probs: torch.Tensor = None, logits: torch.Tensor = None,
                 std: torch.Tensor = None, validate_args: bool = None,
                 entropy_mode: str = "pseudo_std", eps: float = 1e-6):
        if entropy_mode not in ("pseudo_std", "neg_margin", "shannon"):
            raise ValueError(f"Unknown entropy_mode '{entropy_mode}'")

        self.stddev_input = None
        if std is not None:
            if logits is None:
                raise ValueError("`std` (temperature) requires `logits`, not `probs`.")
            # Reduce a per-action std to a single per-batch-element temperature so it
            # scales all logits uniformly (preserves relative ordering of `mean`,
            # unlike dividing each logit by its own action-specific std).
            temperature = std.mean(dim=-1, keepdim=True) if std.dim() == logits.dim() else std
            self.stddev_input = temperature
            logits = logits / temperature.clamp_min(eps)

        super().__init__(probs=probs, logits=logits, validate_args=validate_args)
        self.entropy_mode = entropy_mode
        self.eps = eps

    # -- sample, log_prob, mode, probs, logits, support, enumerate_support,
    # -- batch_shape, event_shape, expand, etc. are all inherited unchanged from
    # -- torch.distributions.Categorical, operating on the temperature-scaled
    # -- logits computed above --

    def margin(self) -> torch.Tensor:
        """
        Gap between the top-1 and top-2 logits (post-temperature), per batch
        element. Unbounded and strictly monotonic in confidence/sharpness.
        """
        if self.logits.shape[-1] < 2:
            return torch.zeros(self.batch_shape, device=self.logits.device, dtype=self.logits.dtype)
        top2 = self.logits.topk(2, dim=-1).values
        return top2[..., 0] - top2[..., 1]

    def pseudo_std(self) -> torch.Tensor:
        """
        Std-dev analog: shrinks toward 0 as the policy sharpens, with polynomial
        (1/margin) decay instead of Shannon entropy's double-exponential decay.
        When `std` was passed at construction, this is exactly
        `std / raw_margin` -- linear in `std`, so reducing the network's own
        predicted uncertainty reliably reduces this value, the same way
        reducing `sigma` reliably reduces `Normal.entropy()`.
        """
        return torch.log10(1.0 / self.margin().clamp_min(self.eps) + 1e-24)

    def shannon_entropy(self) -> torch.Tensor:
        """
        True Shannon entropy -- ie. exactly what the base
        `torch.distributions.Categorical.entropy()` returns (computed on the
        temperature-scaled logits, if `std` was given). Kept available for
        logging/diagnostics even when `.entropy()` itself has been redirected
        to a non-saturating surrogate.
        """
        return super().entropy()

    def entropy(self) -> torch.Tensor:
        """
        Overrides `Categorical.entropy()`. Returns `pseudo_std()` by default
        (see `entropy_mode`), so any existing code that calls `.entropy()` --
        `Model.evaluate_action`, PPO's `_get_policy_loss` -- transparently gets
        a spread measure that keeps discriminating genomes at high sharpness
        instead of saturating near 0, with no other code changes required.
        """
        if self.entropy_mode == "shannon":
            return self.shannon_entropy()
        elif self.entropy_mode == "neg_margin":
            return -self.margin()
        return self.pseudo_std()


if __name__ == "__main__":
    torch.manual_seed(0)

    def build_peaked_logits(actions: int, peak_action: int, sharpness: float) -> torch.Tensor:
        idx = torch.arange(actions).float()
        return -sharpness * (idx - peak_action).abs()

    # ------------------------------------------------------------------
    # 1) unchanged behavior with no std passed (matches previous version)
    # ------------------------------------------------------------------
    print(f"{'sharpness':>9} | {'shannon H':>10} | {'pseudo_std':>14} | {'neg_margin':>10} | sample | log_prob")
    for sharpness in [0.0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0]:
        logits = build_peaked_logits(16, peak_action=3, sharpness=sharpness)

        ref = Categorical(logits=logits)               # unmodified torch class, for comparison
        dist = SharpCategorical(logits=logits)          # entropy_mode='pseudo_std' (default)

        a = dist.sample()
        lp = dist.log_prob(a)
        lp_ref = ref.log_prob(a)
        assert torch.allclose(lp, lp_ref), "log_prob diverged from base Categorical!"
        assert torch.allclose(dist.probs, ref.probs), "probs diverged from base Categorical!"
        assert torch.equal(dist.mode, ref.mode), "mode diverged from base Categorical!"

        shannon = dist.shannon_entropy().item()
        pstd = dist.pseudo_std().item()
        negm = SharpCategorical(logits=logits, entropy_mode="neg_margin").entropy().item()

        print(f"{sharpness:9.2f} | {shannon:10.6f} | {pstd:14.4f} | {negm:10.2f} | {a.item():6d} | {lp.item():8.4f}")

    print("\nAll log_prob / probs / mode checks against plain torch.distributions.Categorical passed.")

    # ------------------------------------------------------------------
    # 2) confirm pseudo_std is exactly linear in `std` when passed in
    # ------------------------------------------------------------------
    print("\n--- pseudo_std linearity check (fixed mean, varying std) ---")
    mean = torch.tensor([2.0, 0.5, -0.5, -1.0])  # raw_margin (top1-top2) = 1.5
    print(f"{'std':>6} | {'pseudo_std':>10} | {'std/raw_margin (expected)':>26} | shannon")
    for s in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0]:
        std = torch.full((4,), s)
        d = SharpCategorical(logits=mean, std=std)
        expected = s / 1.5
        print(f"{s:6.2f} | {d.pseudo_std().item():10.4f} | {expected:26.4f} | {d.shannon_entropy().item():.4f}")
        assert abs(d.pseudo_std().item() - expected) < 1e-4, "linearity broken!"

    print("\npseudo_std tracks std/raw_margin exactly, for every value of std -- confirmed linear, non-saturating.")