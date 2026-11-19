#!/usr/bin/env bash
# Start TensorBoard in the background, then keep the container alive for `docker exec`.
set -e
LOGDIR="${TB_LOGDIR:-/workspace/minimind/tb_logs}"
mkdir -p "$LOGDIR"
tensorboard --logdir "$LOGDIR" --bind_all --port 6006 >> "$LOGDIR/tensorboard.out" 2>&1 &
exec sleep infinity
