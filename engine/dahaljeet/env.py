"""Single-agent-view environment wrapper for RL training.

The learner controls ONE team (both its seats); the opposing team is driven by
a fixed opponent policy. Transitions are recorded per decision of the learning
team, with reward assigned by the chosen scheme.

Kept dependency-free so the engine still imports without numpy/torch.
"""

from __future__ import annotations

import random

from .encode import encode, legal_mask
from .hand import Hand, team_of
from .rewards import SCHEMES, Snapshot
from .view import make_view


class DahalJeetEnv:
    def __init__(self, opponent, reward="potential", learner_team=0,
                 dealer=0, rng=None):
        self.opponent = opponent
        self.reward_fn = SCHEMES[reward]
        self.reward_name = reward
        self.learner_team = learner_team
        self.dealer = dealer
        self.rng = rng or random.Random()
        self.hand = None
        self._prev = None

    def reset(self):
        self.hand = Hand(dealer=self.dealer, rng=self.rng)
        self.hand.deal()
        self._prev = Snapshot(self.hand)
        return self._advance_to_learner()

    def _advance_to_learner(self):
        """Let the opponent act until it is the learner's turn (or the hand ends)."""
        h = self.hand
        while not h.is_over and team_of(h.to_act) != self.learner_team:
            h.play(self.opponent.act(make_view(h, h.to_act)))
        if h.is_over:
            return None
        v = make_view(h, h.to_act)
        return encode(v), legal_mask(v)

    def step(self, card):
        h = self.hand
        h.play(card)
        obs = self._advance_to_learner()
        done = h.is_over
        result = h.result() if done else None
        cur = Snapshot(h)
        r = self.reward_fn(self._prev, cur, self.learner_team, done, result)
        self._prev = cur
        return obs, r, done, result

    def legal(self):
        v = make_view(self.hand, self.hand.to_act)
        return list(v.legal)
