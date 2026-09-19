"""Perfect-information (double-dummy) solver for Dahal Jeet.

Every win rate we report is relative to other agents. None of them says how far
from OPTIMAL any agent is, because we have no ceiling. This computes one: with
all four hands visible and both sides playing optimally, does a given side win?

    python perfect_info.py --verify          # rules match the engine, solver is exact
    python perfect_info.py --probe           # how deep can we actually solve?
    python perfect_info.py --endgame 5 --deals 300

WHY THE VALUE IS A BOOLEAN

The hand is won on the tens: 3+ tens wins, <=1 loses, and 2-2 is decided on
trick count (rules W1-W5). That condition is monotone in (tens, tricks) but it
is NOT the same as maximising tens -- with the tens split 2-2 an optimal line
may concede a ten to secure the seventh trick. So the solver optimises the WIN
INDICATOR directly. That is also what makes it tractable: the value set is
{0,1}, so boolean minimax short-circuits, which IS alpha-beta with the tightest
possible window.

CUTOFFS (all exact, none heuristic)
  * a side holding 3 tens has already won -- no need to play on
  * if a side cannot still reach 2 tens, it has lost
  * once all four tens are gone and split 2-2, the hand reduces to a trick race
    with hard bounds on both sides

HONESTY NOTE
Full 52-ply solving in Python may not be affordable. `--probe` measures it
rather than assuming. If only endgames are affordable, the result must be
reported as an ENDGAME optimality gap and never described as a whole-game
ceiling.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

sys.path.insert(0, ".")

from dahaljeet.cards import is_ten, suit_of
from dahaljeet.hand import Hand, team_of

sys.setrecursionlimit(10000)


# ------------------------------------------------------------------ rules

def legal_from(hand, lead_suit):
    if lead_suit < 0:
        return list(hand)
    follow = [c for c in hand if suit_of(c) == lead_suit]
    return follow if follow else list(hand)


def collapse_equivalents(moves, remaining):
    """Drop moves that are provably interchangeable with a cheaper one.

    Two cards of the same suit held by the same player are equivalent for all
    future play if every card strictly between them has already been played:
    nothing can ever distinguish them in a trick comparison, and swapping them
    is an isomorphism of the remaining game.

    The exception is a TEN. Tens are the entire objective (W1), so a ten is
    never interchangeable with a neighbour -- the isomorphism would change the
    score. This is the one place where a generic trick-taking solver's
    equivalence rule has to be adapted to this game.

    `remaining` must contain every card that can still affect a comparison:
    all cards held by ANY player, PLUS the cards already in the current trick.
    Omitting the trick is a real bug -- with the queen led and the jack and king
    in hand, jack and king are NOT interchangeable even though the queen is no
    longer in anyone's hand. Verification against exhaustive minimax caught
    exactly that case.
    """
    out = []
    for c in sorted(moves):
        if out:
            p = out[-1]
            if (p // 13 == c // 13 and not is_ten(p) and not is_ten(c)
                    and all(x not in remaining for x in range(p + 1, c))):
                continue
        out.append(c)
    return out


def trick_winner(trick, trump, lead_suit):
    best_seat, best_card = trick[0]
    best_trump = suit_of(best_card) == trump
    for seat, card in trick[1:]:
        s = suit_of(card)
        if best_trump:
            if s == trump and card > best_card:
                best_seat, best_card = seat, card
        elif s == trump:
            best_seat, best_card, best_trump = seat, card, True
        elif s == lead_suit and card > best_card:
            best_seat, best_card = seat, card
    return best_seat


# ----------------------------------------------------------------- solver

class Aborted(Exception):
    """A single solve exceeded its node or time budget."""


class Solver:
    def __init__(self, trump, us, max_nodes=None, deadline=None):
        self.trump = trump
        self.us = us            # team index whose win we are computing
        self.memo = {}
        self.nodes = 0
        # A budget checked BETWEEN solves cannot bound a solve that never
        # returns, so the limit is enforced INSIDE the recursion.
        self.max_nodes = max_nodes
        self.deadline = deadline

    def key(self, hands, to_act, trick, lead, tens, tricks):
        return (hands, to_act, trick, lead, tens, tricks)

    def solve(self, hands, to_act, trick, lead, tens, tricks):
        """True iff team `self.us` wins with optimal play by both sides."""
        us, them = self.us, 1 - self.us

        # --- exact terminal cutoffs, checked before any recursion
        if tens[us] >= 3:
            return True
        if tens[them] >= 3:
            return False
        rem_tens = 4 - tens[0] - tens[1]
        if tens[us] + rem_tens <= 1:
            return False
        if tens[them] + rem_tens <= 1:
            return True
        if rem_tens == 0:
            # tens are 2-2: pure trick race, with hard bounds
            rem_tricks = 13 - tricks[0] - tricks[1]
            if tricks[us] >= 7:
                return True
            if tricks[us] + rem_tricks < 7:
                return False

        if not any(hands):
            return tens[us] > tens[them] or (tens[us] == 2 and tricks[us] >= 7)

        k = self.key(hands, to_act, trick, lead, tens, tricks)
        hit = self.memo.get(k)
        if hit is not None:
            return hit

        self.nodes += 1
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise Aborted(f"node cap {self.max_nodes:,} exceeded")
        if self.deadline is not None and (self.nodes & 0xFFF) == 0 \
                and time.perf_counter() > self.deadline:
            raise Aborted("time budget exceeded")
        maximizing = team_of(to_act) == us
        moves = legal_from(hands[to_act], lead)
        if len(moves) > 1:
            remaining = set()
            for hh in hands:
                remaining.update(hh)
            remaining.update(c for _, c in trick)
            moves = collapse_equivalents(moves, remaining)
        # ordering: strongest first when maximising, weakest first when not.
        # Tens are tried early either way -- they are the whole objective.
        moves.sort(key=lambda c: (is_ten(c), suit_of(c) == self.trump, c),
                   reverse=maximizing)

        result = not maximizing
        for card in moves:
            nh = list(hands)
            nh[to_act] = tuple(x for x in nh[to_act] if x != card)
            ntrick = trick + ((to_act, card),)
            nlead = suit_of(card) if lead < 0 else lead

            if len(ntrick) < 4:
                v = self.solve(tuple(nh), (to_act + 1) % 4, ntrick, nlead,
                               tens, tricks)
            else:
                w = trick_winner(ntrick, self.trump, nlead)
                t = team_of(w)
                ntens = list(tens)
                ntricks = list(tricks)
                ntricks[t] += 1
                for _, c in ntrick:
                    if is_ten(c):
                        ntens[t] += 1
                v = self.solve(tuple(nh), w, (), -1,
                               tuple(ntens), tuple(ntricks))

            if maximizing and v:
                result = True
                break
            if not maximizing and not v:
                result = False
                break

        self.memo[k] = result
        return result


def solve_position(h: Hand, us: int, max_nodes=None, time_limit=None):
    """Solve the CURRENT position of a live Hand for team `us`.

    Raises Aborted if the position exceeds the given budget.
    """
    deadline = time.perf_counter() + time_limit if time_limit else None
    s = Solver(h.trump_suit, us, max_nodes=max_nodes, deadline=deadline)
    hands = tuple(tuple(sorted(x)) for x in h.hands)
    trick = tuple(h.trick)
    lead = h.trick_lead_suit if h.trick else -1
    v = s.solve(hands, h.to_act, trick, lead,
                tuple(h.tens_by_team), tuple(h.tricks_by_team))
    return v, s.nodes


# ------------------------------------------------------------ verification

def brute_value(hands, to_act, trick, lead, trump, tens, tricks, us):
    """Reference: exhaustive minimax with NO cutoffs, no memo, no ordering."""
    if not any(hands):
        return tens[us] > tens[1 - us] or (tens[us] == 2 and tricks[us] >= 7)
    maximizing = team_of(to_act) == us
    vals = []
    for card in legal_from(hands[to_act], lead):
        nh = list(hands)
        nh[to_act] = tuple(x for x in nh[to_act] if x != card)
        nt = trick + ((to_act, card),)
        nl = suit_of(card) if lead < 0 else lead
        if len(nt) < 4:
            v = brute_value(tuple(nh), (to_act + 1) % 4, nt, nl, trump,
                            tens, tricks, us)
        else:
            w = trick_winner(nt, trump, nl)
            t = team_of(w)
            a, b = list(tens), list(tricks)
            b[t] += 1
            for _, c in nt:
                if is_ten(c):
                    a[t] += 1
            v = brute_value(tuple(nh), w, (), -1, trump, tuple(a), tuple(b), us)
        vals.append(v)
        if maximizing and v:
            break
        if not maximizing and not v:
            break
    return any(vals) if maximizing else all(vals)


def verify(n_rules=300, n_solver=40, seed=0, depth=9):
    print("VERIFICATION\n")
    ok = True
    rng = random.Random(seed)

    # 1. my rule primitives must agree with the engine, move for move
    mismatches = 0
    for _ in range(n_rules):
        h = Hand(dealer=rng.randrange(4), rng=rng)
        h.deal()
        while not h.is_over:
            mine = sorted(legal_from(tuple(h.hands[h.to_act]),
                                     h.trick_lead_suit if h.trick else -1))
            theirs = sorted(h.legal_moves(h.to_act))
            if mine != theirs:
                mismatches += 1
                break
            if len(h.trick) == 3:
                pre = tuple(h.trick) + ((h.to_act, mine[0]),)
                want = trick_winner(pre, h.trump_suit,
                                    h.trick_lead_suit if h.trick else suit_of(mine[0]))
                got = h.play(mine[0])
                if want != got:
                    mismatches += 1
                    break
            else:
                h.play(rng.choice(mine))
    r1 = mismatches == 0
    ok &= r1
    print(f"  [{'PASS' if r1 else 'FAIL'}] legal moves and trick winners match "
          f"the engine over {n_rules} deals ({mismatches} mismatches)")

    # 2. cutoff/memo solver must equal exhaustive minimax on small endgames
    diff = 0
    tested = 0
    for _ in range(n_solver):
        h = Hand(dealer=rng.randrange(4), rng=rng)
        h.deal()
        while not h.is_over and h.tricks_played < depth:
            h.play(rng.choice(h.legal_moves(h.to_act)))
        if h.is_over:
            continue
        hands = tuple(tuple(sorted(x)) for x in h.hands)
        lead = h.trick_lead_suit if h.trick else -1
        want = brute_value(hands, h.to_act, tuple(h.trick), lead, h.trump_suit,
                           tuple(h.tens_by_team), tuple(h.tricks_by_team), 0)
        got, _ = solve_position(h, 0)
        tested += 1
        if bool(want) != bool(got):
            diff += 1
    r2 = diff == 0
    ok &= r2
    print(f"  [{'PASS' if r2 else 'FAIL'}] solver (with equivalence collapsing) "
          f"equals exhaustive minimax on {tested} {13-depth}-trick endgames "
          f"({diff} disagreements)")

    # 3. the two teams cannot both win the same position
    both = 0
    for _ in range(20):
        h = Hand(dealer=rng.randrange(4), rng=rng)
        h.deal()
        while not h.is_over and h.tricks_played < 10:
            h.play(rng.choice(h.legal_moves(h.to_act)))
        if h.is_over:
            continue
        v0, _ = solve_position(h, 0)
        v1, _ = solve_position(h, 1)
        if v0 == v1:
            both += 1
    r3 = both == 0
    ok &= r3
    print(f"  [{'PASS' if r3 else 'FAIL'}] exactly one team wins each solved "
          f"position ({both} contradictions)")

    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


# ------------------------------------------------------------------ probe

def probe(max_tricks=13, per_depth=8, per_solve=20.0, seed=1):
    """Measure how deep full solving is affordable, with a HARD per-solve cap.

    Each position gets `per_solve` seconds. Positions that exceed it are
    recorded as unsolved rather than allowed to run unbounded -- an earlier
    version checked the budget only between solves, so one deep position could
    run for hours and the gate never fired.
    """
    print(f"FEASIBILITY PROBE (hard cap {per_solve:.0f}s per position)\n",
          flush=True)
    rng = random.Random(seed)
    rows = []
    for k in range(1, max_tricks + 1):
        stop_at = 13 - k
        times, nodes, aborted = [], [], 0
        for _ in range(per_depth):
            h = Hand(dealer=rng.randrange(4), rng=rng)
            h.deal()
            while not h.is_over and h.tricks_played < stop_at:
                h.play(rng.choice(h.legal_moves(h.to_act)))
            if h.is_over:
                continue
            t0 = time.perf_counter()
            try:
                _, nd = solve_position(h, 0, time_limit=per_solve)
                times.append(time.perf_counter() - t0)
                nodes.append(nd)
            except Aborted:
                aborted += 1
        n_try = len(times) + aborted
        if n_try == 0:
            continue
        mean_t = sum(times) / len(times) if times else float("inf")
        # SURVIVORSHIP BIAS. The mean is over SOLVED positions only, so once
        # some positions abort it is no longer an estimate of the cost of a
        # typical position -- only the easy ones finished. At 13 tricks the
        # conditional mean can look SMALLER than at 10 simply because only the
        # trivially-cutoff deals completed. The honest summary is the censored
        # lower bound: aborted positions cost at least the cap.
        lower_bound = (sum(times) + aborted * per_solve) / n_try
        row = {"tricks_remaining": k, "n_attempted": n_try,
               "n_solved": len(times), "n_aborted": aborted,
               "solved_fraction": round(len(times) / n_try, 3),
               "mean_seconds_SOLVED_ONLY": round(mean_t, 4) if times else None,
               "mean_seconds_censored_lower_bound": round(lower_bound, 4),
               "max_seconds": round(max(times), 4) if times else None,
               "mean_nodes_solved_only": int(sum(nodes) / len(nodes)) if nodes else None,
               "biased": aborted > 0}
        rows.append(row)
        if times:
            flag = "  <-- BIASED, easy positions only" if aborted else ""
            print(f"  {k:2d} tricks left  {mean_t:8.3f}s mean(solved)  "
                  f">={lower_bound:7.3f}s true mean  "
                  f"{int(sum(nodes)/len(nodes)):>11,} nodes"
                  f"  solved {len(times)}/{n_try}{flag}", flush=True)
        else:
            print(f"  {k:2d} tricks left  ALL {n_try} positions exceeded "
                  f"{per_solve:.0f}s", flush=True)
        if aborted == n_try:
            print(f"\n  stopping: depth {k} is not affordable at this budget",
                  flush=True)
            break
    return rows



# --------------------------------------------------------- endgame study

def endgame_study(agent_name, opp_name, k_tricks, deals, seed,
                  per_solve=30.0, source_name=None):
    """How often does an agent convert a theoretically won endgame?

    For each deal both agents play normally until `k_tricks` remain. The
    position is then solved exactly, giving the double-dummy verdict under
    optimal play by BOTH sides. The agents then finish the hand themselves and
    we compare.

    `source_name` controls WHERE THE POSITIONS COME FROM, and it matters:

      None          each agent plays its own way up to the handover, so every
                    agent is scored on a different distribution of positions.
                    Conversion rates are then NOT comparable across agents --
                    a strong agent may simply arrive at easier endgames.
      <agent name>  a single fixed policy plays all four seats up to the
                    handover, so EVERY evaluated agent faces the identical set
                    of positions and conversion is directly comparable. The
                    theoretical win rate is then identical across agents, which
                    doubles as a check that the protocol did what it claims.

    Both are reported. The common-position run is the comparable one.

    This is an ENDGAME result and must be described as one. It is not a
    whole-game ceiling: play before the solved position was not optimal, so the
    positions reached are not those an optimal player would have reached.

    Reported:
      theoretical_win_rate  fraction of reached positions that are DD wins
      conversion            P(actually won | theoretically won)
      steal                 P(actually won | theoretically lost)  -- opponent error
      gap                   theoretical - actual
    """
    from dahaljeet.agents import REGISTRY
    from dahaljeet.view import make_view

    def factory(spec):
        """Accept a REGISTRY name, or PIMC:<worlds> / ISMCTS:<iters>."""
        if not isinstance(spec, str):
            return spec
        if spec.startswith("PIMC:"):
            from dahaljeet.search import PIMCAgent
            w = int(spec.split(":")[1])
            return lambda rng=None: PIMCAgent(worlds=w, rng=rng)
        if spec.startswith("ISMCTS:"):
            from dahaljeet.search import ISMCTSAgent
            it = int(spec.split(":")[1])
            return lambda rng=None: ISMCTSAgent(iterations=it, rng=rng)
        return REGISTRY[spec]

    A = factory(agent_name)
    B = factory(opp_name)
    S = factory(source_name) if source_name else None
    rng = random.Random(seed)
    base = random.Random(seed).randrange(1 << 30)
    stop_at = 13 - k_tricks

    n = theo = actual = conv_num = conv_den = steal_num = steal_den = 0
    aborted = 0
    for i in range(deals):
        for rot in range(4):
            a = A(rng=random.Random(1000 + i))
            b = B(rng=random.Random(2000 + i))
            seats = {s: (a if ((s - rot) % 4) % 2 == 0 else b) for s in range(4)}
            us = team_of(rot)          # the team `a` occupies this rotation

            h = Hand(dealer=0, rng=random.Random(base + i))
            h.deal()
            if S is not None:
                # one fixed policy drives ALL seats to the handover, so the
                # position set is identical for every agent evaluated
                src = S(rng=random.Random(3000 + i))
                while not h.is_over and h.tricks_played < stop_at:
                    h.play(src.act(make_view(h, h.to_act)))
            else:
                while not h.is_over and h.tricks_played < stop_at:
                    h.play(seats[h.to_act].act(make_view(h, h.to_act)))
            if h.is_over:
                continue
            try:
                v, _ = solve_position(h, us, time_limit=per_solve)
            except Aborted:
                aborted += 1
                continue
            while not h.is_over:
                h.play(seats[h.to_act].act(make_view(h, h.to_act)))
            res = h.result()
            won = (res.winning_team == us)

            n += 1
            theo += int(v)
            actual += int(won)
            if v:
                conv_den += 1
                conv_num += int(won)
            else:
                steal_den += 1
                steal_num += int(won)

    if n == 0:
        return None
    return {"agent": agent_name if isinstance(agent_name, str) else "custom",
            "opponent": opp_name, "position_source": source_name or "own play",
            "tricks_remaining": k_tricks,
            "positions": n, "aborted": aborted,
            "theoretical_win_rate": round(theo / n, 4),
            "actual_win_rate": round(actual / n, 4),
            "gap": round((theo - actual) / n, 4),
            "conversion": round(conv_num / conv_den, 4) if conv_den else None,
            "conversion_n": conv_den,
            "steal": round(steal_num / steal_den, 4) if steal_den else None,
            "steal_n": steal_den}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--tricks", type=int, default=4,
                    help="endgame depth (tricks remaining) for the brute-force check")
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--rules", type=int, default=300)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--endgame", type=int, default=0,
                    help="run the endgame study with this many tricks remaining")
    ap.add_argument("--deals", type=int, default=150)
    ap.add_argument("--opponent", default="TensThenTricks")
    ap.add_argument("--agents", default="Random,GreedyTricks,TenAware,TensThenTricks")
    ap.add_argument("--source", default="",
                    help="fixed policy that generates the positions, so every "
                         "agent is scored on the identical position set")
    ap.add_argument("--budget", type=float, default=20.0,
                    help="hard wall-clock cap per solved position")
    a = ap.parse_args()

    if a.verify:
        sys.exit(0 if verify(n_rules=a.rules, n_solver=a.cases,
                             depth=13 - a.tricks) else 1)
    if a.endgame:
        import runenv
        t0 = time.perf_counter()
        out = []
        print(f"ENDGAME STUDY — exact double-dummy at {a.endgame} tricks "
              f"remaining, {a.deals} deals x4 rotations\n", flush=True)
        sources = [None] + ([a.source] if a.source else [])
        for src in sources:
            print(f"  --- positions from: {src or 'each agent\'s own play'} ---",
                  flush=True)
            for name in a.agents.split(","):
                r = endgame_study(name.strip(), a.opponent, a.endgame, a.deals,
                                  seed=4242, source_name=src)
                if r is None:
                    continue
                out.append(r)
                print(f"    {name.strip():16s} theoretical {r['theoretical_win_rate']:.4f}  "
                      f"actual {r['actual_win_rate']:.4f}  gap {r['gap']:+.4f}  "
                      f"convert {r['conversion']}  steal {r['steal']}  "
                      f"(n={r['positions']})", flush=True)
        el = time.perf_counter() - t0
        json.dump({"env": runenv.snapshot(), "tricks_remaining": a.endgame,
                   "deals": a.deals, "opponent": a.opponent, "results": out,
                   "seconds": el,
                   "scope_note": ("ENDGAME result. Positions are the ones each "
                                  "agent's own play reaches, then solved "
                                  "exactly. This is NOT a whole-game optimality "
                                  "ceiling and must not be described as one.")},
                  open(f"perfect_info_endgame_k{a.endgame}.json", "w"), indent=2)
        print(f"\n{el:.0f}s -> perfect_info_endgame_k{a.endgame}.json")
        sys.exit(0)
    if a.probe:
        rows = probe(per_solve=a.budget)
        import runenv
        json.dump({"env": runenv.snapshot(), "probe": rows,
                   "note": ("Wall-clock to solve one position exactly, by "
                            "tricks remaining. Determines whether a whole-game "
                            "ceiling is affordable or only an endgame one."),
                   "bias_warning": (
                       "Where n_aborted > 0 the solved-only mean is subject to "
                       "SURVIVORSHIP BIAS and can fall as depth increases, "
                       "because only easy positions finish. Use "
                       "mean_seconds_censored_lower_bound. Crucially this also "
                       "rules out computing a whole-game ceiling on 'the deals "
                       "that solve quickly': solvability correlates with the "
                       "outcome (a position where one side reaches three tens "
                       "early cuts off immediately), so that subset is biased "
                       "toward decisive deals and its win rate would not "
                       "estimate the true ceiling.")},
                  open("perfect_info_probe.json", "w"), indent=2)
        print("\n-> perfect_info_probe.json")
