"""MAPPO-style CTDE self-play (Phase 4c).

Dahal Jeet has FIXED PARTNERSHIPS (rule S1): seats 1&3 play against 2&4 for
the whole session. That makes it a genuinely cooperative problem rather than a
free-for-all -- and because table talk is not permitted, partners cannot signal.
Coordination has to be learned from the play itself, which is what makes
centralised training worth testing here.

Centralised Training, Decentralised Execution:
  * ACTOR sees one seat's PlayerView only (360 dims) -- legal at execution time
  * CRITIC sees BOTH partners' observations concatenated (720 dims) -- available
    only in training, which is the whole point of CTDE
  * Both partner seats SHARE the actor's parameters, so experience from either
    seat trains one policy. Combined with seat-relative encoding this means the
    policy transfers across all four seats.

Opponent is a frozen snapshot of the policy, refreshed periodically (self-play),
rather than a fixed heuristic. Training against a strong fixed opponent from
scratch gives very sparse wins; self-play supplies a curriculum automatically.
"""
from __future__ import annotations
import argparse, json, random, sys, time
sys.path.insert(0, '.')
import runenv
import torch, torch.nn as nn, torch.nn.functional as F

from dahaljeet.agents import REGISTRY, Agent
from dahaljeet.encode import encode, legal_mask, OBS_SIZE, ACTION_SIZE
from dahaljeet.hand import Hand, team_of, PARTNER
from dahaljeet.rewards import SCHEMES, Snapshot
from dahaljeet.tournament import duplicate_match
from dahaljeet.view import make_view

DEV = torch.device("cpu")
NEG = -1e9


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(OBS_SIZE, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, ACTION_SIZE))

    def forward(self, x):
        return self.f(x)


class Critic(nn.Module):
    """Centralised: sees both partners' observations."""

    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(OBS_SIZE * 2, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


class PolicyAgent(Agent):
    def __init__(self, actor, name="MAPPO", greedy=True, rng=None):
        super().__init__(rng)
        self.actor, self.name, self.greedy = actor, name, greedy

    @torch.no_grad()
    def act(self, v):
        x = torch.tensor(encode(v), dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(legal_mask(v), dtype=torch.float32).unsqueeze(0)
        lg = self.actor(x).masked_fill(m == 0, NEG)
        if self.greedy:
            return int(lg.argmax(-1).item())
        return int(torch.distributions.Categorical(logits=lg).sample().item())


def play_episode(actor, opponent, reward_fn, learner_team, dealer, rng):
    """One hand. Returns transitions for the learning team's two seats."""
    h = Hand(dealer=dealer, rng=rng)
    h.deal()
    prev = Snapshot(h)
    traj = []
    # joint observation needs both partners' views at the same instant
    while not h.is_over:
        seat = h.to_act
        v = make_view(h, seat)
        if team_of(seat) != learner_team:
            h.play(opponent.act(v))
            continue
        obs = encode(v)
        mask = legal_mask(v)
        pv = make_view(h, PARTNER[seat])
        joint = obs + encode(pv)
        ot = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        mt = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            lg = actor(ot).masked_fill(mt == 0, NEG)
            dist = torch.distributions.Categorical(logits=lg)
            a = dist.sample()
            lp = float(dist.log_prob(a).item())
        h.play(int(a.item()))
        cur = Snapshot(h)
        done = h.is_over
        r = reward_fn(prev, cur, learner_team, done,
                      h.result() if done else None)
        prev = cur
        traj.append([obs, joint, mask, int(a.item()), lp, r, done])
    return traj, h.result()


def train(episodes, reward, seed, log, refresh):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    actor, critic = Actor().to(DEV), Critic().to(DEV)
    frozen = Actor().to(DEV)
    frozen.load_state_dict(actor.state_dict())
    opt = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    reward_fn = SCHEMES[reward]
    gamma, lam, clip, epochs, horizon = 0.99, 0.95, 0.2, 4, 2048
    buf = []
    heur = REGISTRY["TensThenTricks"](rng=random.Random(seed + 3))

    for ep in range(episodes):
        # 80% self-play against a frozen snapshot, 20% against the heuristic so
        # the policy does not drift into a self-play-only equilibrium.
        opp = (PolicyAgent(frozen, greedy=False, rng=rng)
               if rng.random() < 0.8 else heur)
        traj, _ = play_episode(actor, opp, reward_fn, ep % 2, ep % 4, rng)
        buf.extend(traj)

        if len(buf) >= horizon:
            obs = torch.tensor([t[0] for t in buf], dtype=torch.float32)
            jnt = torch.tensor([t[1] for t in buf], dtype=torch.float32)
            msk = torch.tensor([t[2] for t in buf], dtype=torch.float32)
            act = torch.tensor([t[3] for t in buf], dtype=torch.long)
            old = torch.tensor([t[4] for t in buf], dtype=torch.float32)
            rew = [t[5] for t in buf]
            dn = [t[6] for t in buf]
            with torch.no_grad():
                val = critic(jnt).tolist()
            adv, gae = [0.0] * len(buf), 0.0
            for i in reversed(range(len(buf))):
                nv = 0.0 if dn[i] else (val[i + 1] if i + 1 < len(buf) else 0.0)
                delta = rew[i] + gamma * nv - val[i]
                gae = delta + gamma * lam * (0.0 if dn[i] else gae)
                adv[i] = gae
            advt = torch.tensor(adv, dtype=torch.float32)
            ret = advt + torch.tensor(val, dtype=torch.float32)
            advt = (advt - advt.mean()) / (advt.std() + 1e-8)
            for _ in range(epochs):
                lg = actor(obs).masked_fill(msk == 0, NEG)
                dist = torch.distributions.Categorical(logits=lg)
                ratio = (dist.log_prob(act) - old).exp()
                l1, l2 = ratio * advt, torch.clamp(ratio, 1-clip, 1+clip) * advt
                loss = (-torch.min(l1, l2).mean()
                        + 0.5 * F.mse_loss(critic(jnt), ret)
                        - 0.01 * dist.entropy().mean())
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(
                    list(actor.parameters()) + list(critic.parameters()), 0.5)
                opt.step()
            buf = []

        if (ep + 1) % refresh == 0:
            frozen.load_state_dict(actor.state_dict())

        if (ep + 1) % max(1, episodes // 10) == 0:
            a = PolicyAgent(actor, "MAPPO")
            out = {}
            for on in ("Random", "TensThenTricks"):
                r = duplicate_match(a, REGISTRY[on](rng=random.Random(4)),
                                    n_deals=150, seed=999, timed=False)
                out[on] = round(r.win_rate_a, 4)
            print(f"    ep {ep+1:6d} {out}", flush=True)
            log.append({"ep": ep + 1, **out})
    return actor


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15000)
    ap.add_argument("--reward", default="potential")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", type=int, default=500)
    a = ap.parse_args()
    print(f"[MAPPO CTDE self-play | reward={a.reward} | {a.episodes} eps]", flush=True)
    t0 = time.perf_counter(); log = []
    actor = train(a.episodes, a.reward, a.seed, log, a.refresh)
    ag = PolicyAgent(actor, "MAPPO")
    final = {}
    for on in ("Random", "TensThenTricks", "Tuned" if "Tuned" in REGISTRY else "GreedyTricks"):
        if on not in REGISTRY:
            continue
        r = duplicate_match(ag, REGISTRY[on](rng=random.Random(4)),
                            n_deals=600, seed=999, timed=False)
        final[on] = round(r.win_rate_a, 4)
    el = time.perf_counter() - t0
    print(f"FINAL mappo/{a.reward}: {final}  ({el:.0f}s)", flush=True)
    torch.save(actor.state_dict(), f"rl_mappo_{a.reward}_s{a.seed}.pt")
    json.dump({"env": runenv.snapshot(), "algo": "mappo", "reward": a.reward, "episodes": a.episodes,
               "final": final, "curve": log, "seconds": el},
              open(f"rl_mappo_{a.reward}_s{a.seed}.json", "w"), indent=2)
    print("saved", flush=True)
