"""Network, agent wrapper and evaluation shared by every learning script.

These lived in train_rl.py, which meant trainer.py had to import from the
script it was meant to replace. Splitting them out removes that cycle so
train_rl.py can delegate its training loops to trainer.py instead of keeping a
second copy of them -- two copies of a DQN in one paper is a defect waiting to
happen.

train_rl.py re-exports these names, so `from train_rl import Net, TorchAgent`
keeps working everywhere it is already used.
"""
from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")
import torch
import torch.nn as nn

from dahaljeet.agents import Agent, REGISTRY
from dahaljeet.encode import ACTION_SIZE, OBS_SIZE, encode, legal_mask
from dahaljeet.tournament import duplicate_match

DEV = torch.device("cpu")
NEG = -1e9


class Net(nn.Module):
    def __init__(self, out=ACTION_SIZE, critic=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_SIZE, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU())
        self.head = nn.Linear(256, out)
        self.v = nn.Linear(256, 1) if critic else None

    def forward(self, x):
        h = self.body(x)
        return (self.head(h), self.v(h).squeeze(-1)) if self.v is not None \
            else self.head(h)


class TorchAgent(Agent):
    """Wraps a trained net so it can enter the normal tournament harness."""

    def __init__(self, net, name, greedy=True, rng=None):
        super().__init__(rng)
        self.net, self.name, self.greedy = net, name, greedy

    @torch.no_grad()
    def act(self, v):
        x = torch.tensor(encode(v), dtype=torch.float32, device=DEV).unsqueeze(0)
        m = torch.tensor(legal_mask(v), dtype=torch.float32, device=DEV).unsqueeze(0)
        out = self.net(x)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits.masked_fill(m == 0, NEG)
        if self.greedy:
            return int(logits.argmax(dim=-1).item())
        p = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(p, 1).item())


def evaluate(agent, deals=200, seed=999):
    """Duplicate-deal win rate vs Random and vs the strongest heuristic."""
    out = {}
    for opp_name in ("Random", "TensThenTricks"):
        opp = REGISTRY[opp_name](rng=random.Random(4))
        r = duplicate_match(agent, opp, n_deals=deals, seed=seed, timed=False)
        out[opp_name] = round(r.win_rate_a, 4)
    return out


