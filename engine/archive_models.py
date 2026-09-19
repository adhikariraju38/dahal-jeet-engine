"""Collect every trained model into an archive for public release.

Models are currently scattered as `rl_*.pt`, `alphadj_*.pt`, `br_*.pt` in the
working directory. For release they need to be in one place, each paired with
the configuration that produced it, the results it achieved, and a checksum so
anyone can verify they have the same file we evaluated.

    python archive_models.py            # build ../models/
    python archive_models.py --verify   # re-check every checksum

Design points that matter for a release:
  * SHA-256 per file, so a reviewer can confirm the artifact matches the paper
  * every model carries its training config and its measured results
  * a machine-readable manifest.json AND a human-readable README
  * nothing is deleted from the working directory -- files are copied
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

ARCHIVE = "../models"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def classify(fname):
    """Map a checkpoint filename to (family, method, description)."""
    n = os.path.basename(fname)
    if n.startswith("br_"):
        target = n[3:].rsplit("_s", 1)[0]
        return ("exploiter", f"best-response vs {target}",
                f"DQN trained solely to exploit {target}. Used for the "
                f"exploitability analysis, not as a general agent.")
    if n.startswith("alphadj"):
        return ("neural search", "AlphaDJ",
                "Policy+value network used as prior and leaf evaluator inside "
                "determinized information-set search.")
    if "distill" in n:
        return ("distillation", "Distilled ISMCTS",
                "Supervised imitation of ISMCTS. Search-quality play at "
                "roughly 0.1 ms per decision.")
    if "nfsp" in n:
        return ("equilibrium", "NFSP",
                "Neural Fictitious Self-Play. The released network is the "
                "AVERAGE policy, which is the agent.")
    if "deepcfr" in n:
        return ("equilibrium", "Deep CFR",
                "Average-strategy network. No equilibrium guarantee in a "
                "4-player partnership game -- a baseline, not a solution.")
    if "mappo" in n:
        return ("multi-agent RL", "MAPPO (CTDE)",
                "Shared actor across both partner seats; trained with a "
                "centralised critic and self-play.")
    for k, label in (("dqn", "DQN"), ("ppo", "PPO"), ("a2c", "A2C"),
                     ("double_dueling", "Double+Dueling DQN"),
                     ("double", "Double DQN"), ("dueling", "Dueling DQN")):
        if k in n:
            return ("single-agent RL", label,
                    f"{label} agent. Reward scheme is recorded in `config`.")
    return ("other", n, "")


def sidecar_results(stem):
    """Find the JSON produced alongside a checkpoint, if any."""
    for cand in (f"{stem}.json", f"rl_{stem}.json",
                 stem.replace("br_", "exploit_") + ".json"):
        if os.path.exists(cand):
            try:
                return json.load(open(cand))
            except Exception:
                pass
    return None


def build():
    os.makedirs(ARCHIVE, exist_ok=True)
    os.makedirs(f"{ARCHIVE}/checkpoints", exist_ok=True)
    entries = []
    files = sorted(glob.glob("*.pt"))
    if not files:
        print("no .pt files found — nothing to archive")
        return

    for f in files:
        stem = f[:-3]
        family, method, desc = classify(f)
        res = sidecar_results(stem)
        dst = f"{ARCHIVE}/checkpoints/{f}"
        shutil.copy2(f, dst)
        e = {
            "file": f"checkpoints/{f}",
            "family": family,
            "method": method,
            "description": desc,
            "bytes": os.path.getsize(f),
            "sha256": sha256(f),
            "modified": time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(os.path.getmtime(f))),
        }
        if res:
            e["config"] = {k: v for k, v in res.items()
                           if k in ("method", "algo", "reward", "episodes",
                                    "iters", "traversals", "eta", "seed",
                                    "sims", "epochs")}
            if "final" in res:
                e["results"] = res["final"]
            for k in ("best_response_win", "generic_baseline",
                      "exploitability", "generic_agent"):
                if k in res:
                    e.setdefault("results", {})[k] = res[k]
        entries.append(e)
        print(f"  {f:34s} {e['bytes']/1e6:6.1f} MB  {method}")

    manifest = {
        "project": "Dahal Jeet — traditional Terai-Madhesh card game",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine_commit": None,
        "obs_size": 360,
        "action_size": 52,
        "note": ("All checkpoints are PyTorch state_dicts. Architectures are "
                 "defined in the engine source; see README.md for the mapping "
                 "from each file to its class."),
        "count": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "models": entries,
    }
    json.dump(manifest, open(f"{ARCHIVE}/manifest.json", "w"), indent=2)
    write_readme(manifest)
    print(f"\n  {len(entries)} models, "
          f"{manifest['total_bytes']/1e6:.1f} MB -> {ARCHIVE}/")


def write_readme(m):
    by_family = {}
    for e in m["models"]:
        by_family.setdefault(e["family"], []).append(e)

    lines = [
        "# Trained Models — Dahal Jeet",
        "",
        f"{m['count']} checkpoints, {m['total_bytes']/1e6:.1f} MB total. "
        f"Built {m['built']}.",
        "",
        "Released alongside the paper so every reported number can be "
        "reproduced without retraining.",
        "",
        "## Verifying you have the right files",
        "",
        "```bash",
        "cd 05-engine && python archive_models.py --verify",
        "```",
        "",
        "Every checkpoint carries a SHA-256 in `manifest.json`. If a checksum "
        "does not match, the file is not the one the paper evaluated.",
        "",
        "## Loading",
        "",
        "```python",
        "import torch, sys; sys.path.insert(0, '05-engine')",
        "from train_rl import Net, TorchAgent",
        "",
        "net = Net()",
        "net.load_state_dict(torch.load('checkpoints/rl_distill_s0.pt'))",
        "net.eval()",
        "agent = TorchAgent(net, 'Distilled ISMCTS')",
        "```",
        "",
        "Architectures by file:",
        "",
        "| Pattern | Class | Module |",
        "|---|---|---|",
        "| `rl_dqn_*`, `rl_distill_*`, `br_*` | `Net` | `train_rl.py` |",
        "| `rl_ppo_*` | `Net(critic=True)` | `train_rl.py` |",
        "| `rl_double*`, `rl_dueling*` | `DuelNet` | `train_more.py` |",
        "| `rl_mappo_*` | `Actor` | `train_mappo.py` |",
        "| `rl_nfsp_*` | `mlp(52)` | `nfsp.py` |",
        "| `rl_deepcfr_*` | `mlp()` | `deepcfr.py` |",
        "| `alphadj_*` | `PolicyValueNet` | `alphazero.py` |",
        "",
        "`final_tournament.py` also has a loader that tries each architecture "
        "until one fits.",
        "",
    ]
    for fam in sorted(by_family):
        lines += [f"## {fam.title()}", ""]
        for e in sorted(by_family[fam], key=lambda x: x["file"]):
            lines.append(f"### `{os.path.basename(e['file'])}` — {e['method']}")
            if e["description"]:
                lines.append(e["description"])
            if e.get("config"):
                cfg = ", ".join(f"`{k}={v}`" for k, v in e["config"].items()
                                if v is not None)
                lines.append(f"\n**Config:** {cfg}")
            if e.get("results"):
                r = ", ".join(f"{k} {v}" for k, v in e["results"].items())
                lines.append(f"\n**Measured:** {r}")
            lines.append(f"\n`{e['bytes']/1e6:.1f} MB` · "
                         f"`sha256:{e['sha256'][:16]}…`\n")
    lines += [
        "## Licence and citation",
        "",
        "Code is released under the MIT licence; the models, data and figures "
        "under Creative Commons Attribution 4.0 International (CC BY 4.0).",
        "",
        "If you use these, please cite the paper; see CITATION.cff in the "
        "repository root.",
        "",
        "## Provenance",
        "",
        "Every model was trained against the engine in `05-engine/`, which is "
        "tested to the rule specification in `01-rulebooks/` — 31 spec tests, "
        "one per rule ID, plus 7 tests proving no agent can observe hidden "
        "information.",
        "",
    ]
    open(f"{ARCHIVE}/README.md", "w").write("\n".join(lines))


def verify():
    path = f"{ARCHIVE}/manifest.json"
    if not os.path.exists(path):
        print("no manifest — run without --verify first")
        return 1
    m = json.load(open(path))
    bad = 0
    for e in m["models"]:
        p = os.path.join(ARCHIVE, e["file"])
        if not os.path.exists(p):
            print(f"  MISSING  {e['file']}")
            bad += 1
            continue
        got = sha256(p)
        if got != e["sha256"]:
            print(f"  MISMATCH {e['file']}")
            bad += 1
    print(f"\n  {len(m['models']) - bad}/{len(m['models'])} verified"
          + ("" if not bad else f"  — {bad} PROBLEM(S)"))
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    sys.exit(verify() if a.verify else (build() or 0))
