#!/bin/zsh
# Two defects found in the completed run, fixed here.
#
#  1. ENCODING ABLATION was run at 40k episodes while the headline agents train
#     at 120k, and it came out INVERTED: `minimal` (241 features zeroed) beat
#     the full encoding 0.4244 to 0.3469 with non-overlapping CIs. That may be
#     real -- 40k may be too few episodes to exploit a 360-dim input -- but it
#     does not describe the regime we report, so it is re-run at 120k with 3
#     seeds. The 40k artifacts are preserved in ablation_40k/ because the
#     budget comparison is itself a result.
#
#  2. EXPLOITABILITY was uninformative: the DQN oracle was weaker than a plain
#     heuristic, so every value came out negative and bounded nothing. Replaced
#     by a search that simulates the target's true policy for the opponent
#     seats, compared against the SAME search with no opponent model.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=.venv/bin/python
LOG=run_fixes.log
: > $LOG
echo "started $(date)" >> $LOG

echo "=== best-response search (exploitability, proper oracle) ==="
$PY best_response_search.py --deals 300 --iterations 200 >> $LOG 2>&1 &
BR=$!

ABLATE_PIDS=()
echo "=== encoding ablation at 120k episodes, 3 seeds ==="
for s in 0 1 2; do
  for b in none no_ten_status no_voids no_history no_scalars minimal; do
    while (( $(jobs -rp | wc -l) >= 7 )); do sleep 10; done
    echo "[$(date +%H:%M:%S)] START ablate $b seed $s" >> $LOG
    $PY ablate_encoding.py --block $b --episodes 120000 --seed $s >> $LOG 2>&1 &
    ABLATE_PIDS+=($!)
  done
  # Wait only for THIS seed's ablation jobs. A bare `wait` also waits for the
  # best-response search launched above, which stalled the whole ablation
  # behind a 3-hour job and left six cores idle.
  for pid in $ABLATE_PIDS; do wait $pid; done
  ABLATE_PIDS=()
  echo "[$(date +%H:%M:%S)] seed $s complete" >> $LOG
done
wait $BR 2>/dev/null

echo "=== regenerate dependent outputs ===" >> $LOG
$PY make_figures.py    >> $LOG 2>&1
$PY record_compute.py  >> $LOG 2>&1
echo "COMPLETE $(date)" >> $LOG
