#!/bin/zsh
# ============================================================================
#  Dahal Jeet — final research run
#
#  Differences from run_full.sh, each one traceable to a failure it caused:
#
#   * ASSERTS ARTIFACTS. run_full.sh ran every step with `>> $LOG` and never
#     checked anything. Its stage 2 crashed one second in (missing checkpoints)
#     and the run continued for 20 hours looking healthy.
#   * RESUMES. Steps whose artifact already exists and validates are skipped,
#     so the 24 stage-3 runs and 5 stage-4 runs already on disk are reused.
#     They are reusable because the refactor and the buffer optimisation were
#     proven to reproduce them bit-for-bit (optimization_equivalence.json,
#     reproduce_dqn_potential_s0.json).
#   * VERIFIES BEFORE SPENDING. Stage 0 runs every self-test first. There is no
#     point spending 30 hours feeding a solver that disagrees with brute force.
#   * ONE INTERPRETER. run_full.sh used system python3 for stage 1 and the venv
#     everywhere else.
#   * DEEP CFR IS FEASIBLE. The old depth-30 external sampling had not finished
#     6 of 40 iterations in 12.75 hours (~4^26 per traversal). Now depth-limited
#     with rollout evaluation: 528 s/iteration measured, three seeds concurrent.
#   * MAX=7 on 10 logical cores (8 performance). Workers are single-threaded.
# ============================================================================
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=.venv/bin/python
MAX=7
LOG=run_final.log
SEEDS=(0 1 2)
: > $LOG

CURRENT_STAGE="init"
fail(){ echo "" | tee -a $LOG; echo "!!!!!!!!!! ABORT in $CURRENT_STAGE: $1" | tee -a $LOG; exit 1; }
throttle(){ while (( $(jobs -rp | wc -l) >= MAX )); do sleep 5; done }
launch(){ throttle; echo "[$(date +%H:%M:%S)] START $*" >> $LOG; "$@" >> $LOG 2>&1 & }
stage(){ CURRENT_STAGE="$1"; echo "" >>$LOG
         echo "########## $1  $(date '+%F %T') ##########" >>$LOG
         echo "=== $1  $(date '+%H:%M')"; }
step(){ echo "[$(date +%H:%M:%S)] STEP $*" >> $LOG; "$@" >> $LOG 2>&1 || fail "step failed: $*"; }

# --- resume helpers -------------------------------------------------------
# A step is skipped only when its artifact exists AND is non-empty. Anything
# weaker (e.g. "the file is there") would let a truncated file pass.
have(){ [[ -s $1 ]] }
step_if(){ local art=$1; shift
  if have $art; then echo "  [skip] $art" | tee -a $LOG; else step "$@"; fi }
launch_if(){ local art=$1; shift
  if have $art; then echo "  [skip] $art" | tee -a $LOG; else launch "$@"; fi }

expect_files(){ local want=$1; shift
  local -a found; found=( ${~^@}(N) )
  (( ${#found} >= want )) || fail "expected >= $want matching '$*', found ${#found}"
  for f in $found; do [[ -s $f ]] || fail "artifact is empty: $f"; done
  echo "  [ok] ${#found} artifact(s) matching '$*'" | tee -a $LOG }
expect_json(){ [[ -s $1 ]] || fail "missing or empty: $1"
  $PY -c "import json,sys; json.load(open(sys.argv[1]))" $1 || fail "not valid JSON: $1"
  echo "  [ok] $1" | tee -a $LOG }

echo "run started $(date)" >> $LOG
echo "env: OMP_NUM_THREADS=$OMP_NUM_THREADS MAX_WORKERS=$MAX" >> $LOG

# ---------------------------------------------------------------- STAGE 0
stage "0/9  verification — every self-test before any compute is spent"
step $PY perfect_info.py --verify --tricks 5 --cases 25 --rules 100
step $PY analyse_tournament.py --selftest
step $PY complexity.py --selftest
step $PY trainer.py --equivalence-test
step_if optimization_equivalence.json $PY verify_optimization.py --episodes 600

# ---------------------------------------------------------------- STAGE 1
stage "1/9  heuristics: metaheuristic tuning + championship"
step_if tuned_genome.json        $PY tune.py
step_if championship_results.json $PY championship.py
expect_json tuned_genome.json
expect_json championship_results.json

# ---------------------------------------------------------------- STAGE 2
stage "2/9  search: PIMC + ISMCTS latency/strength curve"
step_if eval_curve.json $PY eval_final.py --part curve
expect_json eval_curve.json

# ---------------------------------------------------------------- STAGE 3
stage "3/9  reinforcement learning, 3 seeds x 4 rewards x 2 algos"
for s in $SEEDS; do
  for rew in terminal ten_shaped potential trick_shaped; do
    launch_if rl_dqn_${rew}_s${s}.pt $PY train_rl.py --algo dqn --reward $rew --episodes 120000 --seed $s
    launch_if rl_ppo_${rew}_s${s}.pt $PY train_rl.py --algo ppo --reward $rew --episodes 120000 --seed $s
  done
done
wait
expect_files 24 'rl_dqn_*_s*.pt' 'rl_ppo_*_s*.pt'

# ---------------------------------------------------------------- STAGE 4
stage "4/9  DQN variants, A2C, MAPPO, NFSP, Deep CFR — 3 seeds"
for s in $SEEDS; do
  launch_if rl_double_s${s}.pt         $PY train_more.py  --method double         --episodes 60000 --seed $s
  launch_if rl_double_dueling_s${s}.pt $PY train_more.py  --method double_dueling --episodes 60000 --seed $s
  launch_if rl_a2c_s${s}.pt            $PY train_more.py  --method a2c            --episodes 60000 --seed $s
  launch_if rl_mappo_potential_s${s}.pt $PY train_mappo.py --reward potential --episodes 200000 --seed $s
  launch_if rl_nfsp_s${s}.pt           $PY nfsp.py        --episodes 80000 --seed $s
done
wait
# Deep CFR: all three seeds concurrently. 528 s/iteration measured at depth 10,
# so ~6 h wall-clock for all three rather than ~18 h run one after another.
for s in $SEEDS; do
  launch_if rl_deepcfr_s${s}.pt $PY deepcfr.py --iters 40 --traversals 120 --depth 10 --seed $s
done
wait
expect_files 3 'rl_double_s*.pt'
expect_files 3 'rl_double_dueling_s*.pt'
expect_files 3 'rl_a2c_s*.pt'
expect_files 3 'rl_mappo_*_s*.pt'
expect_files 3 'rl_nfsp_s*.pt'
expect_files 3 'rl_deepcfr_s*.pt'

# ---------------------------------------------------------------- STAGE 5
stage "5/9  distillation + neural-guided search, 3 seeds"
step_if expert_data.pkl $PY gen_expert_data.py --hands 4000 --iters 200 --procs 6
expect_files 1 'expert_data.pkl'
for s in $SEEDS; do launch_if rl_distill_s${s}.pt $PY train_more.py --method distill --epochs 30 --seed $s; done
wait
expect_files 3 'rl_distill_s*.pt'
for s in $SEEDS; do
  step_if alphadj_s${s}.pt $PY alphazero.py --stage value --hands 6000 --epochs 12 --seed $s
  step_if alphadj_eval_s${s}_n100.json $PY alphazero.py --stage eval --sims 100 --seed $s
done
expect_files 3 'alphadj_s*.pt'

# ---------------------------------------------------------------- STAGE 6
stage "6/9  exploitability + encoding ablation + hyperparameter sweep"
for t in TensThenTricks Tuned GreedyTricks Adaptive; do
  launch_if exploit_${t}_s0.json $PY exploit.py --target $t --episodes 50000 --seed 0
done
wait
expect_files 4 'exploit_*.json'
for b in none no_ten_status no_voids no_history no_scalars minimal; do
  launch_if ablate_${b}_s0.json $PY ablate_encoding.py --block $b --episodes 40000 --seed 0
done
wait
expect_files 6 'ablate_*_s0.json'
step_if sweep_s0.json $PY sweep.py --episodes 20000 --seed 0
expect_files 1 'sweep_*.json'

# ---------------------------------------------------------------- STAGE 7
stage "7/9  controls, complexity, determinization ablation, endgame bound"
step_if control_untrained.json $PY control_untrained.py --deals 400 --seeds 3
expect_json control_untrained.json
step_if complexity.json $PY complexity.py --hands 3000
expect_json complexity.json
# 400 deals and per-deal vectors: every CI overlapped at 200, and both arms play
# the same deals, so the paired test is what makes this decidable.
step_if void_constraint_ablation.json $PY ablate_determinize.py --deals 400
expect_json void_constraint_ablation.json
step_if matched_time.json $PY matched_time.py --measure --deals 400
expect_json matched_time.json
step_if perfect_info_probe.json $PY perfect_info.py --probe --budget 30
expect_json perfect_info_probe.json
# --source gives every agent the IDENTICAL position set, so conversion rates
# are comparable rather than confounded with which endgames each agent reaches.
step_if perfect_info_endgame_k7.json $PY perfect_info.py --endgame 7 --deals 300 \
     --agents Random,GreedyTricks,TenAware,Adaptive,TensThenTricks,PIMC:16,ISMCTS:200 \
     --source TensThenTricks
expect_json perfect_info_endgame_k7.json

# ---------------------------------------------------------------- STAGE 7b
stage "7b/9  self-play + opponent-generalisation matrix"
for s in $SEEDS; do
  for opp in Random GreedyTricks TenAware Adaptive TensThenTricks self league; do
    launch_if gen_ppo_${opp}_s${s}.pt $PY generalise.py --train --algo ppo --opponent $opp --episodes 120000 --seed $s
  done
  wait
  for opp in Random GreedyTricks TenAware Adaptive TensThenTricks self league; do
    launch_if gen_dqn_${opp}_s${s}.pt $PY generalise.py --train --algo dqn --opponent $opp --episodes 120000 --seed $s
  done
  wait
done
expect_files 42 'gen_*.pt'
step_if generalisation_matrix.json $PY generalise.py --matrix --deals 600 --deals-search 150
expect_json generalisation_matrix.json

# ---------------------------------------------------------------- STAGE 8
stage "8/9  final evaluation and round-robin"
step_if eval_rl.json $PY eval_final.py --part rl
expect_json eval_rl.json
step $PY final_tournament.py --deals 300
expect_json final_tournament.json
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
