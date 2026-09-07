#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/IMTalker
PY=/workspace/preprocess_5090/bin/python
TRAIN="$ROOT/tools/train_unitalk_all_layers_ddp.py"
DATA=/workspace/personaplex_frontend_adapter_dataset
BASE=/workspace/exps/unitalk_lookahead096_future048_rms50_gb8_e10/checkpoints/epoch_000010.pt
OUT=/workspace/exps/unitalk_causal6_20260629
LOG="$ROOT/logs/unitalk_causal6_20260629"

mkdir -p "$OUT" "$LOG"

common=(
  --dataset_root "$DATA"
  --epochs 20
  --num_workers 2
  --lr 5e-5
  --weight_decay 0.01
  --lambda_mse 1.0
  --lambda_cos 0.1
  --precision bf16
  --rms_dir "$DATA/rms_50hz"
  --silence_weight 3.0
  --resume "$BASE"
  --last_only
)

launch() {
  local gpu=$1
  local name=$2
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 \
    nohup "$PY" "$TRAIN" \
      "${common[@]}" \
      --save_dir "$OUT/$name" \
      "$@" \
      >"$LOG/$name.log" 2>&1 < /dev/null &
  echo "$! $gpu $name" | tee -a "$LOG/pids.txt"
}

rm -f "$LOG/pids.txt"

launch 0 causal_r0 \
  --batch_size 16 \
  --right_context_frames 0 \
  --tail_loss_frames 48 \
  --future_context_frames 0

launch 1 lookahead_r12_240ms \
  --batch_size 16 \
  --right_context_frames 12 \
  --tail_loss_frames 48 \
  --future_context_frames 12

launch 2 lookahead_r24_480ms \
  --batch_size 16 \
  --right_context_frames 24 \
  --tail_loss_frames 48 \
  --future_context_frames 24

launch 3 lookahead_r48_960ms \
  --batch_size 16 \
  --right_context_frames 48 \
  --tail_loss_frames 48 \
  --future_context_frames 48

launch 4 causal_gru2_stateful \
  --batch_size 8 \
  --right_context_frames 0 \
  --recurrent_layers 2 \
  --tail_loss_frames 48 \
  --future_context_frames 0

launch 5 lookahead_r24_consistency \
  --batch_size 8 \
  --right_context_frames 24 \
  --tail_loss_frames 48 \
  --future_context_frames 24 \
  --lambda_velocity 0.1 \
  --lambda_chunk_consistency 0.1 \
  --consistency_crop_frames 48

echo "Logs: $LOG"
echo "Outputs: $OUT"
