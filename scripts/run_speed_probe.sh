#!/usr/bin/env bash
# MiniMind speed probe — run from the host:  bash scripts/run_speed_probe.sh
# Measures steady-state step time at the REAL production batch sizes, with and without
# torch.compile, using the real trainers on the 3000-row sample files. ~10-20 min, GPU required.
# Outputs: experiments/probe/<cfg>.log, TensorBoard runs under tb_logs/probe-*/,
#          a summary table (s/step, tok/s, projected stage ETA, compile speedup).
# Probe weights/checkpoints are deleted at the end; logs and TB events are kept.
# Run one config at a time (sequential below). Do not run while a model server uses the GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p experiments/probe experiments/probe_out

echo "GPU processes right now (training competes with any serving):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
read -r -p "Continue? [y/N] " ans
[ "$ans" = "y" ] || exit 1

run() { # name trainer data bs acc compile
  local name=$1 trainer=$2 data=$3 bs=$4 acc=$5 comp=$6
  echo "=== $name (batch $bs, acc $acc, compile $comp) start $(date +%H:%M:%S) ==="
  docker exec minimind bash -c "cd /workspace/minimind/trainer && python $trainer \
    --use_moe 1 --max_seq_len 768 --data_path ../experiments/train_data/$data \
    --epochs 1 --batch_size $bs --accumulation_steps $acc --use_compile $comp \
    --log_interval 5 --save_interval 100000 --save_weight probe_$name \
    --save_dir ../experiments/probe_out --from_weight none \
    --use_wandb --wandb_project probe-$name \
    > ../experiments/probe/$name.log 2>&1; echo exit=\$? >> ../experiments/probe/$name.log"
  echo "    $(tail -1 "experiments/probe/$name.log")  end $(date +%H:%M:%S)"
}

# Pretrain shape: batch 32 x acc 8 (94 microbatch steps on the sample).
run pt-b32          train_pretrain.py sample_pretrain_3000.jsonl 32 8 0
run pt-b32-compile  train_pretrain.py sample_pretrain_3000.jsonl 32 8 1
# SFT shape: batch 16 x acc 1 (188 steps on the sample), from scratch (no pretrain weight yet).
run sft-b16         train_full_sft.py sample_sft_3000.jsonl      16 1 0
run sft-b16-compile train_full_sft.py sample_sft_3000.jsonl      16 1 1

echo '=== SUMMARY ==='
docker exec minimind bash -c 'cd /workspace/minimind && python scripts/summarize_speed_probe.py'

echo '=== CLEANUP (probe weights/checkpoints only; logs and TB events kept) ==='
rm -f checkpoints/probe_* experiments/probe_out/probe_*
echo 'done.'
