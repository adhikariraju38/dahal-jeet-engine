#!/bin/zsh
# Long-horizon training. The earlier 9k-episode runs were far too short to
# judge RL on -- with a perfect, fast simulator there is no reason not to run
# orders of magnitude longer.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$(dirname "$0")"

echo "=== launching long runs $(date) ==="
.venv/bin/python train_rl.py    --algo dqn  --reward potential  --episodes 150000 --seed 0 > long_dqn_potential.log   2>&1 &
.venv/bin/python train_rl.py    --algo dqn  --reward ten_shaped --episodes 150000 --seed 1 > long_dqn_tenshaped.log   2>&1 &
.venv/bin/python train_mappo.py --reward potential --episodes 300000 --refresh 2000     > long_mappo.log           2>&1 &

# wait for the expert-data generator, then distil and run the DQN variants
while pgrep -f "[g]en_expert_data.py" >/dev/null; do sleep 10; done
echo "=== expert data ready, starting distillation $(date) ==="
.venv/bin/python train_more.py --method distill --epochs 25 > long_distill.log 2>&1 &
.venv/bin/python train_more.py --method double_dueling --episodes 60000 > long_doubleduel.log 2>&1 &
.venv/bin/python train_more.py --method double         --episodes 60000 > long_double.log     2>&1 &
.venv/bin/python train_more.py --method a2c            --episodes 60000 > long_a2c.log        2>&1 &
wait
echo "=== ALL LONG RUNS COMPLETE $(date) ==="
