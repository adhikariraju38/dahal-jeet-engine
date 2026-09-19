#!/bin/zsh
# Reward ablation: 4 schemes x 2 algorithms. One thread each to avoid
# oversubscribing 10 cores with 8 concurrent torch processes.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
EP=9000
for algo in dqn ppo; do
  for rew in terminal ten_shaped potential trick_shaped; do
    .venv/bin/python train_rl.py --algo $algo --reward $rew \
        --episodes $EP --seed 0 > "ablate_${algo}_${rew}.log" 2>&1 &
  done
done
wait
echo "ABLATION COMPLETE"
