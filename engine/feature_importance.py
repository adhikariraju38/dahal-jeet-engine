"""Permutation feature importance for the state encoding (gap E5).

`ablate_encoding.py` answers a different question: it RETRAINS with a block
zeroed, so it measures whether the block is necessary for learning. This
measures whether the ALREADY-TRAINED network actually relies on it. Both are
worth reporting: a field can be learnable-around (small ablation effect) yet
heavily used by the trained policy, or vice versa.

Method: play matches in which one block of the observation is replaced, at
every decision, with the corresponding slice from a DIFFERENT randomly drawn
game state. That destroys the block's relationship with the current position
while preserving its marginal distribution -- which is what separates
permutation importance from simply zeroing, since zeroing also moves the input
off the data manifold and can hurt for reasons unrelated to relevance.

    python feature_importance.py --deals 300

Reported as the drop in win rate against a fixed opponent. Larger drop = the
policy leans on that block more.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

sys.path.insert(0, ".")
import runenv
import torch

from dahaljeet.agents import REGISTRY
from dahaljeet.encode import (OBS_SIZE, OFF_HAND, OFF_PLAYED, OFF_SCALARS,
                              OFF_TENS, OFF_TRICK, OFF_VOID, SZ_HAND,
                              SZ_PLAYED, SZ_SCALARS, SZ_TENS, SZ_TRICK,
                              SZ_VOID, encode, legal_mask)
from dahaljeet.hand import Hand
from dahaljeet.tournament import duplicate_match
from dahaljeet.view import make_view
from nets import NEG, Net, TorchAgent

BLOCKS = {
    "hand": (OFF_HAND, SZ_HAND),
    "history": (OFF_PLAYED, SZ_PLAYED),
    "ten_status": (OFF_TENS, SZ_TENS),
    "current_trick": (OFF_TRICK, SZ_TRICK),
    "voids": (OFF_VOID, SZ_VOID),
    "scalars": (OFF_SCALARS, SZ_SCALARS),
}


def build_pool(n_hands, seed):
    """A pool of real observations to draw replacement slices from."""
    rng = random.Random(seed)
    pool = []
    for _ in range(n_hands):
        h = Hand(dealer=rng.randrange(4), rng=rng)
        h.deal()
        while not h.is_over:
            pool.append(encode(make_view(h, h.to_act)))
            h.play(rng.choice(h.legal_moves(h.to_act)))
    return pool


class PermutedAgent(TorchAgent):
    """Plays normally except one block is drawn from an unrelated state."""

    def __init__(self, net, name, block, pool, rng):
        super().__init__(net, name, greedy=True, rng=rng)
        self.block = block
        self.pool = pool
        self.prng = random.Random(99)

    @torch.no_grad()
    def act(self, v):
        obs = list(encode(v))
        if self.block is not None:
            off, sz = BLOCKS[self.block]
            donor = self.pool[self.prng.randrange(len(self.pool))]
            obs[off:off + sz] = donor[off:off + sz]
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(legal_mask(v), dtype=torch.float32).unsqueeze(0)
        out = self.net(x)
        logits = out[0] if isinstance(out, tuple) else out
        return int(logits.masked_fill(m == 0, NEG).argmax(-1))


def load_any(path):
    sd = torch.load(path, map_location="cpu")
    for critic in (False, True):
        try:
            net = Net(critic=critic)
            net.load_state_dict(sd)
            net.eval()
            return net
        except Exception:
            continue
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--opponent", default="TensThenTricks")
    ap.add_argument("--pool-hands", type=int, default=200)
    ap.add_argument("--models", default="rl_ppo_potential_s0.pt,rl_distill_s0.pt")
    a = ap.parse_args()

    t0 = time.perf_counter()
    pool = build_pool(a.pool_hands, seed=5)
    print(f"replacement pool: {len(pool):,} real observations\n", flush=True)

    rows = []
    for path in a.models.split(","):
        path = path.strip()
        if not os.path.exists(path):
            print(f"  [skip] {path} not on disk")
            continue
        net = load_any(path)
        if net is None:
            print(f"  [skip] {path}: no matching architecture")
            continue
        opp = REGISTRY[a.opponent](rng=random.Random(4))
        base = duplicate_match(
            PermutedAgent(net, "base", None, pool, random.Random(7)),
            opp, n_deals=a.deals, seed=717, timed=False)
        b_lo, b_hi = base.bootstrap_ci()
        print(f"{path}  baseline {base.win_rate_a:.4f} "
              f"[{b_lo:.4f},{b_hi:.4f}]", flush=True)
        entry = {"model": path, "baseline": round(base.win_rate_a, 4),
                 "baseline_ci": [round(b_lo, 4), round(b_hi, 4)], "blocks": {}}
        for blk in BLOCKS:
            opp = REGISTRY[a.opponent](rng=random.Random(4))
            r = duplicate_match(
                PermutedAgent(net, f"perm-{blk}", blk, pool, random.Random(7)),
                opp, n_deals=a.deals, seed=717, timed=False)
            lo, hi = r.bootstrap_ci()
            drop = base.win_rate_a - r.win_rate_a
            entry["blocks"][blk] = {
                "win": round(r.win_rate_a, 4), "ci": [round(lo, 4), round(hi, 4)],
                "drop": round(drop, 4), "features": BLOCKS[blk][1]}
            print(f"    permute {blk:14s} {r.win_rate_a:.4f} "
                  f"[{lo:.4f},{hi:.4f}]   drop {drop:+.4f}  "
                  f"({BLOCKS[blk][1]} features)", flush=True)
        rows.append(entry)

    el = time.perf_counter() - t0
    json.dump({"env": runenv.snapshot(), "deals": a.deals,
               "opponent": a.opponent, "obs_size": OBS_SIZE,
               "method": ("Each block is replaced at every decision by the same "
                          "slice taken from an unrelated real game state. This "
                          "preserves the block's marginal distribution while "
                          "destroying its relationship to the current position, "
                          "unlike zeroing, which also pushes the input off the "
                          "data manifold."),
               "results": rows, "seconds": el},
              open("feature_importance.json", "w"), indent=2)
    print(f"\n{el:.0f}s -> feature_importance.json")
