#!/bin/zsh
# ============================================================================
#  Dahal Jeet — full research run, with complete per-worker environment capture
#
#  Every worker records its own OMP_NUM_THREADS, torch thread count, device and
#  MPS status at save time (runenv.snapshot()), so the compute record states
#  what training actually used rather than what the reporting shell happened to
#  have set.
#
#  MAX=5 concurrent workers: ~3.5 GB of 16 GB. Six pushed this machine into
#  swap during development, and swapping is far slower than queueing.
# ============================================================================
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=.venv/bin/python
MAX=5
LOG=full_run.log
SEEDS=(0 1 2)
: > $LOG

throttle(){ while (( $(jobs -rp | wc -l) >= MAX )); do sleep 5; done }
launch(){ throttle; echo "[$(date +%H:%M:%S)] START $*" >> $LOG; "$@" >> $LOG 2>&1 & }
stage(){ echo "" >>$LOG; echo "########## $1  $(date '+%F %T') ##########" >>$LOG; echo "=== $1"; }

echo "run started $(date)" >> $LOG
echo "env: OMP_NUM_THREADS=$OMP_NUM_THREADS MAX_WORKERS=$MAX" >> $LOG

# ---------------------------------------------------------------- STAGE 1
stage "1/7  heuristics: metaheuristic tuning + championship"
python3 tune.py       >> $LOG 2>&1
python3 championship.py >> $LOG 2>&1

# ---------------------------------------------------------------- STAGE 2
stage "2/7  search: PIMC + ISMCTS latency/strength curve"
$PY eval_final.py --part curve >> $LOG 2>&1   # no models needed

# ---------------------------------------------------------------- STAGE 3
stage "3/7  reinforcement learning, 3 seeds x 4 rewards x 2 algos"
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

# ---------------------------------------------------------------- STAGE 4
stage "4/7  DQN variants, A2C, MAPPO, NFSP, Deep CFR — 3 seeds"
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

# ---------------------------------------------------------------- STAGE 5
stage "5/7  distillation + neural-guided search, 3 seeds"
[[ -f expert_data.pkl ]] || $PY gen_expert_data.py --hands 4000 --iters 200 --procs 5
for s in $SEEDS; do launch $PY train_more.py --method distill --epochs 30 --seed $s; done
wait
for s in $SEEDS; do
  $PY alphazero.py --stage value --hands 6000 --epochs 12 --seed $s >> $LOG 2>&1
  $PY alphazero.py --stage eval  --sims 100 --seed $s               >> $LOG 2>&1
done

# ---------------------------------------------------------------- STAGE 6
stage "6/7  exploitability + encoding ablation + hyperparameter sweep"
for t in TensThenTricks Tuned GreedyTricks Adaptive; do
  launch $PY exploit.py --target $t --episodes 50000 --seed 0
done
wait
for b in none no_ten_status no_voids no_history no_scalars minimal; do
  launch $PY ablate_encoding.py --block $b --episodes 40000 --seed 0
done
wait
$PY sweep.py --episodes 20000 --seed 0 >> $LOG 2>&1

# ---------------------------------------------------------------- STAGE 7
stage "7/7  final tournament, figures, model archive, compute record"
$PY eval_final.py --part rl >> $LOG 2>&1      # needs stage-3 models
$PY final_tournament.py --deals 300 >> $LOG 2>&1
$PY make_figures.py     >> $LOG 2>&1
$PY archive_models.py   >> $LOG 2>&1
$PY archive_models.py --verify >> $LOG 2>&1
$PY record_compute.py   >> $LOG 2>&1

echo "" >> $LOG
echo "########## COMPLETE $(date '+%F %T') ##########" >> $LOG
