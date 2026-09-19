#!/bin/zsh
# ============================================================================
#  Dahal Jeet — full reproduction pipeline
#
#  Regenerates every number and figure in the paper from scratch.
#
#      ./reproduce.sh            # everything (long)
#      ./reproduce.sh stage2     # one stage
#      ./reproduce.sh --list     # what each stage does
#
#  Requires: python3 (stdlib only for stages 0-2), plus numpy+torch in
#  .venv for the learning stages.  Set up with:
#      python3 -m venv .venv && .venv/bin/pip install numpy torch matplotlib
# ============================================================================
set -e
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=.venv/bin/python
SEEDS=(0 1 2)                 # multi-seed: every learning result is 3 seeds
STAGE=${1:-all}

banner(){ echo; echo "============================================================"; echo "  $1"; echo "============================================================"; }

if [[ "$STAGE" == "--list" ]]; then
cat <<'EOF'
  stage0  engine + leakage tests            ~1 min    stdlib only
  stage1  heuristics: round robin + tuning  ~15 min   stdlib only
  stage2  search: PIMC/ISMCTS + latency     ~60 min   stdlib only
  stage3  RL: 6 methods x 4 rewards x 3 seeds  ~10 h   needs torch
  stage4  distillation (expert data + train)   ~40 min needs torch
  stage5  NFSP + Deep CFR                      ~4 h    needs torch
  stage6  AlphaDJ (neural-guided search)       ~1 h    needs torch
  stage7  exploitability (best-response)       ~6 h    needs torch
  stage8  ablations: encoding + hyperparams    ~4 h    needs torch
  stage9  final all-play-all tournament        ~3 h
  stage10 figures + model archive              ~5 min
EOF
exit 0
fi

run_stage(){ [[ "$STAGE" == "all" || "$STAGE" == "$1" ]]; }

# ---------------------------------------------------------------- stage 0
if run_stage stage0; then
banner "STAGE 0 — engine correctness and information-leakage proof"
python3 -m unittest discover -s tests -v 2>&1 | tail -5
fi

# ---------------------------------------------------------------- stage 1
if run_stage stage1; then
banner "STAGE 1 — rule-based agents and metaheuristic tuning"
python3 tune.py
python3 championship.py
fi

# ---------------------------------------------------------------- stage 2
if run_stage stage2; then
banner "STAGE 2 — search agents and the latency/strength curve"
$PY eval_final.py
fi

# ---------------------------------------------------------------- stage 3
if run_stage stage3; then
banner "STAGE 3 — reinforcement learning, 3 seeds per configuration"
for s in $SEEDS; do
  for rew in terminal ten_shaped potential trick_shaped; do
    $PY train_rl.py --algo dqn --reward $rew --episodes 150000 --seed $s &
    $PY train_rl.py --algo ppo --reward $rew --episodes 150000 --seed $s &
  done
  wait
  $PY train_more.py --method double         --episodes 60000 --seed $s &
  $PY train_more.py --method double_dueling --episodes 60000 --seed $s &
  $PY train_more.py --method a2c            --episodes 60000 --seed $s &
  $PY train_mappo.py --reward potential --episodes 300000 --seed $s &
  wait
done
fi

# ---------------------------------------------------------------- stage 4
if run_stage stage4; then
banner "STAGE 4 — expert data generation and policy distillation"
[[ -f expert_data.pkl ]] || $PY gen_expert_data.py --hands 4000 --iters 200 --procs 6
for s in $SEEDS; do $PY train_more.py --method distill --epochs 30 --seed $s; done
fi

# ---------------------------------------------------------------- stage 5
if run_stage stage5; then
banner "STAGE 5 — NFSP and Deep CFR"
for s in $SEEDS; do
  $PY nfsp.py    --episodes 60000 --seed $s &
  $PY deepcfr.py --iters 60 --traversals 200 --seed $s &
  wait
done
fi

# ---------------------------------------------------------------- stage 6
if run_stage stage6; then
banner "STAGE 6 — neural-guided information-set search"
for s in $SEEDS; do
  $PY alphazero.py --stage value --hands 6000 --epochs 12 --seed $s
  $PY alphazero.py --stage eval  --sims 100 --seed $s
done
fi

# ---------------------------------------------------------------- stage 7
if run_stage stage7; then
banner "STAGE 7 — best-response exploitability"
for t in TensThenTricks Tuned GreedyTricks PIMC16 ISMCTS250; do
  $PY exploit.py --target $t --episodes 40000 --seed 0 &
done
wait
fi

# ---------------------------------------------------------------- stage 8
if run_stage stage8; then
banner "STAGE 8 — encoding ablation and hyperparameter sensitivity"
for b in none no_ten_status no_voids no_history no_scalars minimal; do
  $PY ablate_encoding.py --block $b --episodes 30000 --seed 0 &
done
wait
$PY sweep.py --episodes 15000 --seed 0
fi

# ---------------------------------------------------------------- stage 9
if run_stage stage9; then
banner "STAGE 9 — final all-play-all tournament"
$PY final_tournament.py --deals 400
fi

# ---------------------------------------------------------------- stage 10
if run_stage stage10; then
banner "STAGE 10 — figures and model archive"
$PY make_figures.py
$PY archive_models.py
$PY archive_models.py --verify
fi

banner "DONE"
echo "  results  -> 05-engine/*.json"
echo "  figures  -> 08-figures/"
echo "  models   -> 10-models/"
