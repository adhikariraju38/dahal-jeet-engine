#!/bin/zsh
# Remaining work: encoding ablation seeds 1-2, and the best-response search
# with all five targets running CONCURRENTLY.
#
# Two bugs in the previous version, both fixed here:
#  * a bare `wait` after each seed also waited on the best-response search, so
#    the ablation stalled behind a 3-hour job with six cores idle;
#  * killing the orchestrator killed the search with it (same process group),
#    losing two hours of completed targets because the artifact is only written
#    at the end. Each target now writes its own file as it finishes.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=.venv/bin/python
LOG=run_fixes2.log
: > $LOG
echo "started $(date)" >> $LOG

# --- best-response search: one process per target, independent artifacts
BR_PIDS=()
for t in Random GreedyTricks TenAware Adaptive TensThenTricks; do
  echo "[$(date +%H:%M:%S)] START best-response $t" >> $LOG
  nohup $PY best_response_search.py --targets $t --deals 300 --iterations 200 \
        --out exploitability_$t.json >> $LOG 2>&1 &
  BR_PIDS+=($!)
done

# --- encoding ablation, seeds 1 and 2
for s in 1 2; do
  A_PIDS=()
  for b in none no_ten_status no_voids no_history no_scalars minimal; do
    echo "[$(date +%H:%M:%S)] START ablate $b seed $s" >> $LOG
    nohup $PY ablate_encoding.py --block $b --episodes 120000 --seed $s >> $LOG 2>&1 &
    A_PIDS+=($!)
  done
  for pid in $A_PIDS; do wait $pid; done       # this seed only
  echo "[$(date +%H:%M:%S)] seed $s complete" >> $LOG
done

for pid in $BR_PIDS; do wait $pid; done
echo "[$(date +%H:%M:%S)] merging per-target exploitability" >> $LOG
$PY merge_exploitability.py >> $LOG 2>&1
$PY make_figures.py   >> $LOG 2>&1
$PY record_compute.py >> $LOG 2>&1
echo "COMPLETE $(date)" >> $LOG
