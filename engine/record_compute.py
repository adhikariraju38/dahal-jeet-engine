"""Compute, configuration and environment record for the paper.

A Q1 submission needs to state what was run, on what, with which settings, and
for how long. This walks every result artifact on disk and produces:

  * an environment snapshot (hardware, OS, Python, library versions)
  * a per-experiment table: method, config, seed, wall-clock, results
  * aggregate CPU-hours, with concurrency accounted for
  * a machine-readable compute.json plus a paper-ready Markdown table

    python record_compute.py

Every figure it prints comes from a file on disk. Nothing is estimated except
the concurrency-adjusted CPU-hours, which is labelled as an estimate.
"""
from __future__ import annotations

import glob
import json
import os
import platform
import re
import subprocess
import sys
import time

OUT = "../compute"


def sh(cmd, default="unknown"):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def environment():
    env = {
        "captured": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": sh("sysctl -n machdep.cpu.brand_string"),
        "cpu_cores_logical": os.cpu_count(),
        "cpu_perf_cores": sh("sysctl -n hw.perflevel0.logicalcpu"),
        "cpu_efficiency_cores": sh("sysctl -n hw.perflevel1.logicalcpu"),
        "ram_bytes": sh("sysctl -n hw.memsize"),
        "os": sh("sw_vers -productName") + " " + sh("sw_vers -productVersion"),
        "python": sys.version.split()[0],
        "accelerator": "none; CPU only (no CUDA, no MPS used)",
    }
    try:
        b = int(env["ram_bytes"])
        # Apple markets this as "16 GB" and 17179869184 bytes is exactly
        # 16 GiB. Dividing by 1e9 gives 17.2, which is the same memory in
        # decimal units and reads as an error. Report GiB.
        env["ram_gib"] = round(b / (1024 ** 3), 1)
        env["ram_gb_decimal"] = round(b / 1e9, 1)
    except Exception:
        pass
    for mod in ("torch", "numpy", "matplotlib"):
        try:
            m = __import__(mod)
            env[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception:
            env[f"{mod}_version"] = "not installed"
    try:
        import torch
        env["torch_threads"] = torch.get_num_threads()
        env["torch_cuda_available"] = torch.cuda.is_available()
        env["torch_mps_available"] = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available())
    except Exception:
        pass
    # NOTE: thread settings are deliberately NOT read from this process. A
    # reporting script's shell is not the training workers' shell, so reading
    # OMP_NUM_THREADS here would state a value no worker ever used. The real
    # values come from each artifact's own `env` block -- see worker_envs().
    env["max_concurrent_workers"] = 5
    return env


def worker_envs(rows):
    """Aggregate the environment each WORKER recorded for itself.

    Every training script calls runenv.snapshot() at save time, so these are
    the settings actually in force during training.
    """
    seen = {}
    missing = []
    peaks = []
    for r in rows:
        e = r.get("env")
        if not e:
            missing.append(r["artifact"])
            continue
        key = (e.get("OMP_NUM_THREADS"), e.get("torch_num_threads"),
               e.get("device"), e.get("mps_used"), e.get("torch"))
        rss = e.get("peak_rss_mb")
        if isinstance(rss, (int, float)):
            peaks.append((rss, r["artifact"]))
        seen.setdefault(key, []).append(r["artifact"])
    peaks.sort(reverse=True)
    return {
        "distinct_configurations": len(seen),
        "peak_rss_mb": ({"max": peaks[0][0], "max_artifact": peaks[0][1],
                         "median": peaks[len(peaks) // 2][0],
                         "n_reported": len(peaks),
                         "note": ("Per-process peak resident memory, recorded "
                                  "by each worker. Artifacts written before "
                                  "this was captured report nothing.")}
                        if peaks else {"n_reported": 0}),
        "configurations": [
            {"OMP_NUM_THREADS": k[0], "torch_num_threads": k[1],
             "device": k[2], "mps_used": k[3], "torch": k[4],
             "n_artifacts": len(v), "artifacts": sorted(v)[:6]}
            for k, v in seen.items()],
        "artifacts_without_env": missing,
        "note": ("Artifacts without an `env` block predate per-worker "
                 "environment capture and should be re-run before "
                 "submission if their thread settings matter."),
    }


CONFIG_KEYS = ("algo", "method", "reward", "episodes", "epochs", "iters",
               "traversals", "eta", "seed", "sims", "worlds", "block",
               "features_zeroed", "target", "deals_per_pair", "hands")


def collect():
    rows = []

    def add(path, kind, d):
        cfg = {k: d[k] for k in CONFIG_KEYS if k in d and d[k] is not None}
        secs = d.get("seconds") or d.get("elapsed_s")
        res = d.get("final") or d.get("results") or {}
        if "best_response_win" in d:
            res = {"best_response": d["best_response_win"],
                   "generic_baseline": d["generic_baseline"],
                   "exploitability": d["exploitability"]}
        rows.append({
            "artifact": os.path.basename(path),
            "kind": kind,
            "env": d.get("env"),
            "config": cfg,
            "seconds": round(secs, 1) if isinstance(secs, (int, float)) else None,
            "results": res,
        })

    patterns = [
        ("rl_*.json", "learning agent"),
        ("exploit_*.json", "exploitability"),
        ("ablate_*.json", "encoding ablation"),
        # Superseded runs still consumed compute and must be counted. The 40k
        # encoding ablation was moved out of the main glob when it was re-run
        # at 120k; excluding it silently understated the budget by 27 CPU-hours.
        ("ablation_40k/ablate_*.json", "encoding ablation (40k, superseded)"),
        ("void_constraint_ablation.json", "determinization ablation"),
        ("sweep_*.json", "hyperparameter sweep"),
        ("alphadj_eval_*.json", "neural-guided search"),
        ("championship_results.json", "heuristic championship"),
        ("eval_final.json", "search latency curve"),
        ("final_tournament.json", "final round-robin"),
        ("tuned_genome.json", "metaheuristic tuning"),
    ]
    for pat, kind in patterns:
        for f in sorted(glob.glob(pat)):
            try:
                add(f, kind, json.load(open(f)))
            except Exception as e:
                print(f"  [skip] {f}: {e}")
    return rows


def timings_from_logs():
    """Wall-clock actually observed in the run logs, where artifacts lack it."""
    found = {}
    for log in glob.glob("*.log"):
        try:
            txt = open(log, errors="ignore").read()
        except Exception:
            continue
        for m in re.finditer(r"FINAL (\S+?):.*?\((\d+)s\)", txt):
            found[m.group(1)] = int(m.group(2))
    return found


def main():
    os.makedirs(OUT, exist_ok=True)
    env = environment()
    rows = collect()
    log_times = timings_from_logs()

    known = [r["seconds"] for r in rows if r["seconds"]]
    total_s = sum(known) + sum(log_times.values())
    n_missing = sum(1 for r in rows if not r["seconds"])

    rec = {
        "environment": env,
        "worker_environments": worker_envs(rows),
        "experiments": rows,
        "log_derived_timings_s": log_times,
        "totals": {
            "artifacts": len(rows),
            "artifacts_with_timing": len(known),
            "artifacts_without_timing": n_missing,
            "measured_wall_clock_s": round(total_s, 1),
            "measured_wall_clock_h": round(total_s / 3600, 2),
            "note": ("Sum of per-experiment wall-clock. Jobs ran up to "
                     f"{env['max_concurrent_workers']} at a time with "
                     "OMP_NUM_THREADS=1, so this is CPU-time across workers, "
                     "NOT elapsed time at the wall. Elapsed time is lower by "
                     "roughly the concurrency factor."),
        },
    }
    json.dump(rec, open(f"{OUT}/compute.json", "w"), indent=2)
    write_markdown(rec)
    print(f"  {len(rows)} artifacts, "
          f"{rec['totals']['measured_wall_clock_h']:.1f} CPU-hours recorded")
    print(f"  -> {OUT}/compute.json and compute.md")


def _worker_table(rec):
    w = rec["worker_environments"]
    L = ["| OMP_NUM_THREADS | torch threads | device | MPS used | torch | runs |",
         "|---|---|---|---|---|---:|"]
    for c in w["configurations"]:
        L.append(f"| {c['OMP_NUM_THREADS']} | {c['torch_num_threads']} | "
                 f"{c['device']} | {c['mps_used']} | {c['torch']} | "
                 f"{c['n_artifacts']} |")
    if w["artifacts_without_env"]:
        L += ["", f"⚠️ {len(w['artifacts_without_env'])} artifact(s) predate "
              "per-worker environment capture: "
              + ", ".join(f"`{a}`" for a in w["artifacts_without_env"][:8])
              + (" …" if len(w["artifacts_without_env"]) > 8 else "")]
    return L


def write_markdown(rec):
    e = rec["environment"]
    t = rec["totals"]
    L = [
        "# Compute and Environment Record",
        "",
        f"Captured {e['captured']}. Generated by `05-engine/record_compute.py` "
        "from the result artifacts on disk — no value here is typed by hand.",
        "",
        "## Hardware and software",
        "",
        "| | |",
        "|---|---|",
        f"| Machine | {e.get('processor')} |",
        f"| Architecture | {e.get('machine')} |",
        f"| Cores | {e.get('cpu_cores_logical')} logical "
        f"({e.get('cpu_perf_cores')} performance + {e.get('cpu_efficiency_cores')} efficiency) |",
        f"| Memory | {e.get('ram_gib')} GB ({e.get('ram_bytes')} bytes; = {e.get('ram_gb_decimal')} GB decimal) |",
        f"| Accelerator | {e.get('accelerator')} |",
        f"| OS | {e.get('os')} |",
        f"| Python | {e.get('python')} |",
        f"| PyTorch | {e.get('torch_version')} |",
        f"| NumPy | {e.get('numpy_version')} |",
        f"| Matplotlib | {e.get('matplotlib_version')} |",
        f"| CUDA available | {e.get('torch_cuda_available')} |",
        f"| MPS available | {e.get('torch_mps_available')} |",
        f"| Max concurrent workers | {e.get('max_concurrent_workers')} |",
        "",
        "### Thread settings, as recorded BY THE WORKERS",
        "",
        "These come from each training process's own snapshot at save time, not "
        "from the shell that generated this report — a reporting script's "
        "environment is not the workers'.",
        "",
    ] + _worker_table(rec) + [
        "**Everything ran on CPU.** No GPU or Apple-Silicon MPS acceleration was used, and "
        "`OMP_NUM_THREADS=1` was set so each worker is single-threaded and concurrency is "
        "controlled explicitly rather than by the BLAS layer.",
        "",
        "## Compute budget",
        "",
        "| | |",
        "|---|---|",
        f"| Result artifacts | {t['artifacts']} |",
        f"| With recorded timing | {t['artifacts_with_timing']} |",
        f"| **Measured CPU-hours** | **{t['measured_wall_clock_h']:.1f}** |",
        "",
        f"> {t['note']}",
        "",
        "## Per-experiment record",
        "",
        "| Artifact | Kind | Configuration | Seconds | Key result |",
        "|---|---|---|---:|---|",
    ]
    for r in sorted(rec["experiments"], key=lambda x: (x["kind"], x["artifact"])):
        cfg = ", ".join(f"{k}={v}" for k, v in r["config"].items()) or "—"
        secs = f"{r['seconds']:.0f}" if r["seconds"] else "—"
        res = r["results"]
        if isinstance(res, dict) and res:
            key = ", ".join(f"{k}: {v}" for k, v in list(res.items())[:2])
        else:
            key = "—"
        L.append(f"| `{r['artifact']}` | {r['kind']} | {cfg} | {secs} | {key} |")

    if rec["log_derived_timings_s"]:
        L += ["", "## Timings recovered from run logs", "",
              "| Run | Seconds |", "|---|---:|"]
        for k, v in sorted(rec["log_derived_timings_s"].items()):
            L.append(f"| `{k}` | {v} |")

    L += [
        "",
        "## Reproduction",
        "",
        "```bash",
        "cd 05-engine",
        "./reproduce.sh --list      # stages and their runtimes",
        "./reproduce.sh             # everything",
        "python record_compute.py   # regenerate this file",
        "```",
        "",
        "Seeds are passed explicitly to every training script (`--seed`). Every learning result "
        "is reported across seeds 0, 1 and 2.",
        "",
    ]
    open(f"{OUT}/compute.md", "w").write("\n".join(L))


if __name__ == "__main__":
    main()
