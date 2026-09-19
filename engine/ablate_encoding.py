"""State-encoding ablation (gap #8).

We claim the observation is designed so that a learning agent does not have to
infer the objective from raw play history. That claim was never tested: we
never removed a field to show it mattered.

This ablates each block by zeroing it, retrains, and reports the drop. A field
whose removal costs nothing was not earning its place, and saying so is more
honest than asserting the design is good.

Blocks ablated:
  none          full encoding (control)
  no_ten_status per-ten [out / ours / theirs] -- the objective, handed over directly
  no_voids      inferred voids -- public knowledge a human player also uses
  no_history    who has played what so far
  no_scalars    running tens/tricks counts
  minimal       own hand + current trick only

Usage:  python ablate_encoding.py --block no_ten_status --episodes 30000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque

sys.path.insert(0, ".")
import runenv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dahaljeet.agents import REGISTRY
from dahaljeet.encode import (OFF_HAND, SZ_HAND, OFF_PLAYED, SZ_PLAYED,
                              OFF_TENS, SZ_TENS, OFF_TRICK, SZ_TRICK,
                              OFF_VOID, SZ_VOID, OFF_SCALARS, SZ_SCALARS,
                              OBS_SIZE)
from dahaljeet.env import DahalJeetEnv
from dahaljeet.tournament import duplicate_match
from train_rl import Net, TorchAgent

NEG = -1e9
_T0 = 0.0

# index ranges to zero out, per ablation
BLOCKS = {
    "none": [],
    "no_ten_status": [(OFF_TENS, SZ_TENS)],
    "no_voids": [(OFF_VOID, SZ_VOID)],
    "no_history": [(OFF_PLAYED, SZ_PLAYED)],
    "no_scalars": [(OFF_SCALARS, SZ_SCALARS)],
    "minimal": [(OFF_PLAYED, SZ_PLAYED), (OFF_TENS, SZ_TENS),
                (OFF_VOID, SZ_VOID), (OFF_SCALARS, SZ_SCALARS)],
}


def make_mask(block):
    """A 1/0 vector applied elementwise to every observation.

    float32 array rather than a Python list: the mask is applied at every
    decision and every stored transition, and the list version was a 360-element
    Python loop each time. Values are exactly 0.0 or 1.0, so multiplying in
    float32 gives bit-identical results to multiplying in float64 and casting.
    """
    m = np.ones(OBS_SIZE, dtype=np.float32)
    for off, size in BLOCKS[block]:
        m[off:off + size] = 0.0
    return m


class MaskedAgent(TorchAgent):
    """Wraps a trained net so evaluation uses the same ablated view."""

    def __init__(self, net, name, mask, rng=None):
        super().__init__(net, name, greedy=True, rng=rng)
        self.mask = mask

    @torch.no_grad()
    def act(self, v):
        from dahaljeet.encode import encode, legal_mask
        obs = np.asarray(encode(v), dtype=np.float32) * self.mask
        x = torch.from_numpy(obs).unsqueeze(0)
        lm = torch.tensor(legal_mask(v), dtype=torch.float32).unsqueeze(0)
        out = self.net(x)
        logits = out[0] if isinstance(out, tuple) else out
        return int(logits.masked_fill(lm == 0, NEG).argmax(-1))


def train(block, episodes, seed, reward="potential"):
    global _T0
    _T0 = time.perf_counter()
    mask = make_mask(block)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    q, tgt = Net(), Net()
    tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    buf = deque(maxlen=100_000)
    opp = REGISTRY["TensThenTricks"](rng=random.Random(seed + 1))
    eps, gamma, batch, step = 1.0, 0.99, 128, 0

    def ap(o):
        return np.asarray(o, dtype=np.float32) * mask

    for ep in range(episodes):
        env = DahalJeetEnv(opp, reward=reward, learner_team=ep % 2,
                           dealer=ep % 4, rng=rng)
        s = env.reset()
        while s is not None:
            obs, lm = s
            obs = ap(obs)
            legal = [i for i, m in enumerate(lm) if m]
            if rng.random() < eps:
                a = rng.choice(legal)
            else:
                with torch.no_grad():
                    ql = q(torch.from_numpy(np.ascontiguousarray(obs)).unsqueeze(0))
                    a = int(ql.masked_fill(
                        torch.tensor(lm).unsqueeze(0) == 0, NEG).argmax(-1))
            s2, r, done, _ = env.step(a)
            nobs = ap(s2[0]) if s2 is not None else None
            nlm = s2[1] if s2 is not None else None
            buf.append((obs, a, r, nobs, nlm, done))
            s = s2
            step += 1
            if len(buf) >= 2000 and step % 4 == 0:
                bt = rng.sample(buf, batch)
                bo = torch.from_numpy(np.stack([b[0] for b in bt]))
                ba = torch.tensor([b[1] for b in bt], dtype=torch.long)
                target = torch.tensor([b[2] for b in bt], dtype=torch.float32)
                nz = [i for i, b in enumerate(bt)
                      if not b[5] and b[3] is not None]
                if nz:
                    no = torch.from_numpy(np.stack([bt[i][3] for i in nz]))
                    nm = torch.from_numpy(np.stack([np.asarray(bt[i][4], dtype=np.float32)
                                                    for i in nz]))
                    with torch.no_grad():
                        sel = q(no).masked_fill(nm == 0, NEG).argmax(-1, keepdim=True)
                        target[nz] += gamma * tgt(no).gather(1, sel).squeeze(1)
                pred = q(bo).gather(1, ba.unsqueeze(1)).squeeze(1)
                loss = F.smooth_l1_loss(pred, target)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 10.0); opt.step()
            if step % 2000 == 0:
                tgt.load_state_dict(q.state_dict())
        eps = max(0.05, eps * 0.9997)
        # Progress output. A 120k-episode run is ~6 hours; without this it is
        # indistinguishable from a hung process, which is exactly the problem
        # Deep CFR had. Printing only -- no RNG is consumed, so results are
        # unchanged.
        every = max(1, min(5000, episodes // 20))
        if (ep + 1) % every == 0:
            el = time.perf_counter() - _T0
            rate = (ep + 1) / el
            print(f"    [{block}] ep {ep+1:6d}/{episodes} "
                  f"({100*(ep+1)/episodes:4.1f}%)  {el/60:6.1f} min  "
                  f"eta {(episodes-ep-1)/rate/60:6.1f} min", flush=True)
    return q, mask


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--block", required=True, choices=list(BLOCKS))
    ap_.add_argument("--episodes", type=int, default=30000)
    ap_.add_argument("--seed", type=int, default=0)
    a = ap_.parse_args()
    t0 = time.perf_counter()
    zeroed = sum(sz for _, sz in BLOCKS[a.block])
    print(f"[encoding ablation: {a.block} — {zeroed}/{OBS_SIZE} features zeroed]",
          flush=True)
    q, mask = train(a.block, a.episodes, a.seed)
    agent = MaskedAgent(q, f"abl-{a.block}", mask, rng=random.Random(1))
    final = {}
    for opp in ("Random", "TensThenTricks"):
        m = duplicate_match(agent, REGISTRY[opp](rng=random.Random(4)),
                            n_deals=800, seed=909, timed=False)
        lo, hi = m.bootstrap_ci()
        final[opp] = {"win": round(m.win_rate_a, 4),
                      "ci": [round(lo, 4), round(hi, 4)]}
    el = time.perf_counter() - t0
    print(f"FINAL ablate/{a.block}: {final} ({el:.0f}s)", flush=True)
    json.dump({"env": runenv.snapshot(), "block": a.block, "features_zeroed": zeroed,
               "episodes": a.episodes, "seed": a.seed,
               "final": final, "seconds": el},
              open(f"ablate_{a.block}_s{a.seed}.json", "w"), indent=2)
