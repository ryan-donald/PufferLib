#!/usr/bin/env bash
# Run from the PufferLib root after building the native executable.
set -euo pipefail
binary=${1:-./puffer}
steps=${2:-10000000}
mkdir -p build/bomber_experiment
for seed in 42 43 44; do
    "$binary" train --base.seed="$seed" --env.map_seed="$seed" \
        --vec.total_agents=1024 --vec.num_threads=4 --vec.num_buffers=2 \
        --train.minibatch_size=16384 --train.total_timesteps="$steps" \
        --base.eval_episodes=0 --base.run_id="bomber_seed_$seed" \
        --base.checkpoint_dir=build/bomber_experiment/checkpoints \
        --base.log_dir=build/bomber_experiment/logs \
        > "build/bomber_experiment/train_$seed.txt" 2>&1
    checkpoint=$(find "build/bomber_experiment/checkpoints/bomber/bomber_seed_$seed" -name '*.bin' | sort | tail -1)
    test -n "$checkpoint"
    "$binary" eval "$checkpoint" --headless --base.seed=123 \
        --env.map_seed=1000000 --base.eval_episodes=1000 --base.eval_agents=256 \
        --vec.num_threads=2 --vec.num_buffers=2 --train.minibatch_size=4096 \
        > "build/bomber_experiment/eval_$seed.txt" 2>&1
    echo "Completed training and held-out evaluation for seed $seed"
done
