#!/bin/zsh
# ============================================================================
#  Dahal Jeet — full research run, v2
#
#  Differences from v1, all of them consequences of problems v1 actually hit:
#
#  1. EVERY STAGE ASSERTS ITS ARTIFACTS. v1 ran each step with `>> $LOG` and
#     never checked the exit status or the output, so a stage could produce
#     nothing while the run continued looking healthy. That cost a wasted
#     evaluation. Here a stage that does not produce what it promised aborts
#     the run, loudly, naming the stage.
#
#  2. ONE INTERPRETER. v1 ran stage 1 under system `python3` and everything
#     else under the venv. Two interpreters means two library versions in one
#     set of results.
#
#  3. PHASE-2 EXPERIMENTS INCLUDED — perfect-information endgame bound,
#     void-constraint ablation, untrained-network control, complexity, and the
#     post-hoc statistics.
#
#  4. VERIFICATION BEFORE COMPUTE. The self-tests run FIRST. There is no point
#     spending 30 hours to feed a solver that disagrees with brute force.
# ============================================================================
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=.venv/bin/python
MAX=5
LOG=full_run_v2.log
SEEDS=(0 1 2)
: > $LOG

fail(){ echo "" | tee -a $LOG; echo "!!!!!!!!!! ABORT in $CURRENT_STAGE: $1" | tee -a $LOG; exit 1; }

throttle(){ while (( $(jobs -rp | wc -l) >= MAX )); do sleep 5; done }
launch(){ throttle; echo "[$(date +%H:%M:%S)] START $*" >> $LOG; "$@" >> $LOG 2>&1 & }
stage(){ CURRENT_STAGE="$1"; echo "" >>$LOG
         echo "########## $1  $(date '+%F %T') ##########" >>$LOG; echo "=== $1"; }

# run a step and abort if it exits non-zero
step(){ echo "[$(date +%H:%M:%S)] STEP $*" >> $LOG; "$@" >> $LOG 2>&1 \
        || fail "step failed: $*"; }

# expect_files <count> <glob...> — every file must exist and be non-empty
expect_files(){
  local want=$1; shift
  local -a found
  found=( ${~^@}(N) )
  (( ${#found} >= want )) || fail "expected >= $want artifacts matching '$*', found ${#found}"
  for f in $found; do
    [[ -s $f ]] || fail "artifact is empty: $f"
  done
  echo "  [ok] ${#found} artifact(s) matching '$*'" | tee -a $LOG
}

# expect_json <file> — must exist, be non-empty and parse
expect_json(){
  [[ -s $1 ]] || fail "missing or empty: $1"
  $PY -c "import json,sys; json.load(open(sys.argv[1]))" $1 \
      || fail "not valid JSON: $1"
  echo "  [ok] $1" | tee -a $LOG
}

echo "run started $(date)" >> $LOG
echo "env: OMP_NUM_THREADS=$OMP_NUM_THREADS MAX_WORKERS=$MAX" >> $LOG

# ---------------------------------------------------------------- STAGE 0
stage "0/9  verification — self-tests before any compute is spent"
step $PY perfect_info.py --verify --tricks 5 --cases 25 --rules 100
step $PY analyse_tournament.py --selftest
step $PY trainer.py --equivalence-test
step $PY complexity.py --selftest

# ---------------------------------------------------------------- STAGE 1
stage "1/9  heuristics: metaheuristic tuning + championship"
step $PY tune.py
step $PY championship.py
expect_json tuned_genome.json
expect_json championship_results.json

# ---------------------------------------------------------------- STAGE 2
stage "2/9  search: PIMC + ISMCTS latency/strength curve"
step $PY eval_final.py --part curve
expect_json eval_curve.json

# ---------------------------------------------------------------- STAGE 3
stage "3/9  reinforcement learning, 3 seeds x 4 rewards x 2 algos"
for s in $SEEDS; do
  for rew in terminal ten_shaped potential trick_shaped; do
    launch $PY train_rl.py --algo dqn --reward $rew --episodes 120000 --seed $s
  done
  wait
  for rew in terminal ten_shaped potential trick_shaped; do
    launch $PY train_rl.py --algo ppo --reward $rew --episodes 120000 --seed $s
  done
  wait
done
expect_files 24 'rl_dqn_*.json' 'rl_ppo_*.json'
expect_files 24 'rl_dqn_*.pt'   'rl_ppo_*.pt'

# ---------------------------------------------------------------- STAGE 4
stage "4/9  DQN variants, A2C, MAPPO, NFSP, Deep CFR — 3 seeds"
for s in $SEEDS; do
  launch $PY train_more.py  --method double         --episodes 60000 --seed $s
  launch $PY train_more.py  --method double_dueling --episodes 60000 --seed $s
  launch $PY train_more.py  --method a2c            --episodes 60000 --seed $s
  launch $PY train_mappo.py --reward potential --episodes 200000 --seed $s
  launch $PY nfsp.py        --episodes 80000 --seed $s
  wait
  launch $PY deepcfr.py --iters 40 --traversals 120 --seed $s
  wait
done
expect_files 3 'rl_double_s*.pt'
expect_files 3 'rl_double_dueling_s*.pt'
expect_files 3 'rl_a2c_s*.pt'
expect_files 3 'rl_mappo_*_s*.pt'
expect_files 3 'rl_nfsp_s*.pt'
expect_files 3 'rl_deepcfr_s*.pt'

# ---------------------------------------------------------------- STAGE 5
stage "5/9  distillation + neural-guided search, 3 seeds"
[[ -f expert_data.pkl ]] || step $PY gen_expert_data.py --hands 4000 --iters 200 --procs 5
[[ -s expert_data.pkl ]] || fail "expert_data.pkl missing or empty"
for s in $SEEDS; do launch $PY train_more.py --method distill --epochs 30 --seed $s; done
wait
expect_files 3 'rl_distill_s*.pt'
for s in $SEEDS; do
  step $PY alphazero.py --stage value --hands 6000 --epochs 12 --seed $s
  step $PY alphazero.py --stage eval  --sims 100 --seed $s
done
expect_files 3 'alphadj_s*.pt'

# ---------------------------------------------------------------- STAGE 6
stage "6/9  exploitability + encoding ablation + hyperparameter sweep"
for t in TensThenTricks Tuned GreedyTricks Adaptive; do
  launch $PY exploit.py --target $t --episodes 50000 --seed 0
done
wait
expect_files 4 'exploit_*.json'
for b in none no_ten_status no_voids no_history no_scalars minimal; do
  launch $PY ablate_encoding.py --block $b --episodes 40000 --seed 0
done
wait
expect_files 6 'ablate_*.json'
step $PY sweep.py --episodes 20000 --seed 0
expect_files 1 'sweep_*.json'

# ---------------------------------------------------------------- STAGE 7
stage "7/9  Phase-2: controls, complexity, determinization + endgame bound"
step $PY control_untrained.py --deals 400 --seeds 3
expect_json control_untrained.json
step $PY complexity.py --hands 3000
expect_json complexity.json
# 400 deals: every CI overlapped at 200, so nothing was decidable.
# Both arms play the same deals, so the artifact keeps per-deal
# vectors and matched_time.py runs a PAIRED test, which is what
# actually buys the power here.
step $PY ablate_determinize.py --deals 400
expect_json void_constraint_ablation.json
# matched-WALL-CLOCK, actually measured rather than interpolated. The
# constraint costs 14-21% more time under ISMCTS, so matched-samples and
# matched-time can disagree there; the measured arm is the citable one.
step $PY matched_time.py --measure --deals 400
expect_json matched_time.json
step $PY perfect_info.py --probe --budget 30
expect_json perfect_info_probe.json
# --source makes every agent face the IDENTICAL position set, so conversion
# rates are comparable. Without it each agent generates its own positions and a
# strong agent can look better simply by arriving at easier endgames.
step $PY perfect_info.py --endgame 7 --deals 300 \
     --agents Random,GreedyTricks,TenAware,Adaptive,TensThenTricks,PIMC:16,ISMCTS:200 \
     --source TensThenTricks
expect_json perfect_info_endgame_k7.json

# ---------------------------------------------------------------- STAGE 7b
stage "7b/9  self-play + opponent-generalisation matrix (gap C8/C9)"
# 7 training opponents x 2 algos x 3 seeds. This is the single largest stage:
# DQN is ~108 min per run, PPO ~12 min, four at a time -> roughly 10-11 hours.
# It exists because nothing else in the pipeline can distinguish "learned the
# game" from "learned TensThenTricks".
for s in $SEEDS; do
  for opp in Random GreedyTricks TenAware Adaptive TensThenTricks self league; do
    launch $PY generalise.py --train --algo ppo --opponent $opp \
           --episodes 120000 --seed $s
  done
  wait
  for opp in Random GreedyTricks TenAware Adaptive TensThenTricks self league; do
    launch $PY generalise.py --train --algo dqn --opponent $opp \
           --episodes 120000 --seed $s
  done
  wait
done
expect_files 42 'gen_*.pt'
step $PY generalise.py --matrix --deals 600 --deals-search 150
expect_json generalisation_matrix.json

# ---------------------------------------------------------------- STAGE 8
stage "8/9  final evaluation and round-robin"
step $PY eval_final.py --part rl
expect_json eval_rl.json
step $PY final_tournament.py --deals 300
expect_json final_tournament.json
# the round-robin MUST have kept per-deal vectors, or stage 9 has nothing to do
$PY -c "
import json,sys
d=json.load(open('final_tournament.json'))
bad=[p for p in d['pairs'] if not p.get('a_wins_per_deal')]
sys.exit(1 if bad or not d['pairs'] else 0)" \
  || fail "final_tournament.json has no per-deal vectors (P0.1 regression)"
echo "  [ok] per-deal vectors present" | tee -a $LOG

# ---------------------------------------------------------------- STAGE 9
stage "9/9  statistics, figures, archive, compute record"
step $PY analyse_tournament.py
expect_json stats_corrections.json
expect_json ratings.json
expect_json power.json
step $PY aggregate_seeds.py
expect_json seed_summary.json
step $PY make_figures.py
expect_files 8 '../figures/*.png'
step $PY archive_models.py
step $PY archive_models.py --verify
step $PY record_compute.py
expect_json ../compute/compute.json

echo "" >> $LOG
echo "########## COMPLETE $(date '+%F %T') ##########" | tee -a $LOG
