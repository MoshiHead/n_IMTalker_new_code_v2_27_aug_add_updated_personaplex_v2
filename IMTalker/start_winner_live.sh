#!/usr/bin/env bash
set -euo pipefail

# Canonical launcher for the live PersonaPlex + IMTalker winners.
# AH is AJ plus anti-burst audio pacing and is the recommended default.

IMTALKER_DIR="${IMTALKER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$IMTALKER_DIR")}"
VENV_DIR="${VENV_DIR:-/workspace/preprocess_5090}"
VARIANT="${VARIANT:-AH}"
VARIANT="${VARIANT^^}"

pick_existing() {
  local candidate
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf 'Missing required asset. Checked:\n' >&2
  printf '  %s\n' "$@" >&2
  return 1
}

case "$VARIANT" in
  AH)
    SERVER_FILE="$IMTALKER_DIR/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
    DEFAULT_PORT=8998
    DEFAULT_GPU=0
    DEFAULT_CFG=1.13
    DUMP_NAME=typeah_audio_pace
    ;;
  AJ)
    SERVER_FILE="$IMTALKER_DIR/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AJNetworkIso.py"
    DEFAULT_PORT=8999
    DEFAULT_GPU=1
    DEFAULT_CFG=1.15
    DUMP_NAME=typeaj_network_iso
    ;;
  *)
    echo "VARIANT must be AH or AJ, got: $VARIANT" >&2
    exit 2
    ;;
esac

PORT="${PORT:-$DEFAULT_PORT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPU}"
A_CFG_SCALE="${A_CFG_SCALE:-$DEFAULT_CFG}"
NFE="${NFE:-5}"
VOICE_PROMPT="${VOICE_PROMPT:-VARM3.pt}"
ENABLE_COMPILED_BLINK="${ENABLE_COMPILED_BLINK:-0}"
ENABLE_EAGER_BLINK="${ENABLE_EAGER_BLINK:-0}"

EXTRA_RENDER_ARGS=()
if [[ "$ENABLE_COMPILED_BLINK" == "1" && "$ENABLE_EAGER_BLINK" == "1" ]]; then
  echo "Enable only one blink mode: compiled or eager." >&2
  exit 2
fi
if [[ "$ENABLE_COMPILED_BLINK" == "1" || "$ENABLE_EAGER_BLINK" == "1" ]]; then
  [[ "$VARIANT" == "AH" ]] || {
    echo "Blink compositing is supported only with VARIANT=AH." >&2
    exit 2
  }
  BLINK_MOTION_PATH="${BLINK_MOTION_PATH:-$(pick_existing \
    "$PROJECT_ROOT/checkpoints/lora/3robert_audio3_ditto_static_motion.pt" \
    /workspace/hf_assets/lora/3robert_audio3_ditto_static_motion.pt \
    "$IMTALKER_DIR/checkpoints/blink_composite/3robert_audio3_ditto_static_motion.pt")}"
  if [[ "$ENABLE_COMPILED_BLINK" == "1" ]]; then
    SERVER_FILE="$IMTALKER_DIR/experiments/transition_blend/AHCompiledBlinkFused.py"
    EXTRA_RENDER_ARGS=(
      --compile_renderer
      --enable_eye_blink_composite
      --blink_motion_path "$BLINK_MOTION_PATH"
    )
  else
    SERVER_FILE="$IMTALKER_DIR/experiments/transition_blend/AHCompiledRendererTransitionBlend.py"
    EXTRA_RENDER_ARGS=(
      --enable_eye_blink_composite
      --blink_motion_path "$BLINK_MOTION_PATH"
    )
  fi
  export IMTALKER_CACHED_ENGINE="${IMTALKER_CACHED_ENGINE:-1}"
fi

PERSONAPLEX_DIR="${PERSONAPLEX_DIR:-$(pick_existing \
  /workspace/personaplex_bnb4 \
  "$PROJECT_ROOT/checkpoints/personaplex_bnb4")}"
ADAPTER_PATH="${ADAPTER_PATH:-$(pick_existing \
  /workspace/hf_assets/personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt \
  "$PROJECT_ROOT/checkpoints/personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt")}"
SILENCE_HELIUM_PATH="${SILENCE_HELIUM_PATH:-$(pick_existing \
  /workspace/hf_assets/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt \
  /workspace/personaplex_frontend_adapter_dataset/stats/silence_helium_mean.pt \
  "$PROJECT_ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt")}"
DISABLE_LORA="${DISABLE_LORA:-0}"
if [[ "$DISABLE_LORA" == "1" ]]; then
  LORA_GENERATOR_PATH=""
  LORA_ARGS=()
else
  LORA_GENERATOR_PATH="${LORA_GENERATOR_PATH:-$(pick_existing \
    "$PROJECT_ROOT/checkpoints/live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt" \
    "$IMTALKER_DIR/checkpoints/ditto_blink_lora_withaudio_r64_1h_last.ckpt" \
    /workspace/hf_assets/lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt \
    "$PROJECT_ROOT/checkpoints/lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt")}"
  LORA_ARGS=(
    --lora_generator_path "$LORA_GENERATOR_PATH"
    --lora_rank 64
    --lora_alpha 128
    --lora_dropout 0.05
  )
fi

GENERATOR_PATH="${GENERATOR_PATH:-$IMTALKER_DIR/checkpoints/generator.ckpt}"
RENDERER_PATH="${RENDERER_PATH:-$IMTALKER_DIR/checkpoints/renderer.ckpt}"
WAV2VEC_MODEL_PATH="${WAV2VEC_MODEL_PATH:-$IMTALKER_DIR/checkpoints/wav2vec2-base-960h}"
REF_PATH="${REF_PATH:-$IMTALKER_DIR/assets/3robert.jpeg}"
PROMPT_FILE="${PROMPT_FILE:-$IMTALKER_DIR/prompts/RB_Robert_System_Prompt_full.txt}"
HTML_PATH="${HTML_PATH:-$IMTALKER_DIR/static/index_v3_binary_fullscreen_aj_nodrop.html}"

for required in \
  "$SERVER_FILE" "$GENERATOR_PATH" "$RENDERER_PATH" \
  "$ADAPTER_PATH" "$SILENCE_HELIUM_PATH" "$REF_PATH" "$PROMPT_FILE" "$HTML_PATH" \
  "$PERSONAPLEX_DIR/model_bnb_4bit.pt" \
  "$PERSONAPLEX_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" \
  "$PERSONAPLEX_DIR/tokenizer_spm_32k_3.model"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

if [[ "${IMTALKER_CACHED_ENGINE:-0}" == "1" ]]; then
  [[ -f "$IMTALKER_DIR/liveTry_cached.py" ]] || {
    echo "Cached engine requested but missing: $IMTALKER_DIR/liveTry_cached.py" >&2
    exit 1
  }
fi

if [[ -z "${VOICE_PROMPT_DIR:-}" ]]; then
  for candidate in \
    "$PERSONAPLEX_DIR/voices" \
    /workspace/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices \
    /root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices \
    "$HOME"/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices; do
    if [[ -f "$candidate/$VOICE_PROMPT" ]]; then
      VOICE_PROMPT_DIR="$candidate"
      break
    fi
  done
fi
[[ -f "${VOICE_PROMPT_DIR:-}/$VOICE_PROMPT" ]] || {
  echo "Cannot find $VOICE_PROMPT. Set VOICE_PROMPT_DIR explicitly." >&2
  exit 1
}

source "$VENV_DIR/bin/activate"
cd "$IMTALKER_DIR"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$IMTALKER_DIR:$PERSONAPLEX_DIR/moshi:$PERSONAPLEX_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export IMTALKER_PROMPT_STATE_CACHE="${IMTALKER_PROMPT_STATE_CACHE:-1}"

ROBERT_PROMPT="$(tr '\n' ' ' < "$PROMPT_FILE")"
DUMP_DIR="${DUMP_DIR:-$IMTALKER_DIR/live_dumps_${DUMP_NAME}_${PORT}_varm3}"

echo "Starting $VARIANT on port $PORT, physical GPU $CUDA_VISIBLE_DEVICES"
echo "Voice: $VOICE_PROMPT | CFG: $A_CFG_SCALE | NFE: $NFE"
[[ "$DISABLE_LORA" == "1" ]] && echo "Generator: base checkpoint only (LoRA disabled)"
if [[ "$ENABLE_COMPILED_BLINK" == "1" ]]; then
  echo "Renderer: compiled fused blink | motion: $BLINK_MOTION_PATH"
  echo "PersonaPlex prompt cache engine: ${IMTALKER_CACHED_ENGINE:-0}"
elif [[ "$ENABLE_EAGER_BLINK" == "1" ]]; then
  echo "Renderer: eager blink (torch.compile disabled) | motion: $BLINK_MOTION_PATH"
  echo "PersonaPlex prompt cache engine: ${IMTALKER_CACHED_ENGINE:-0}"
fi

exec python -u "$SERVER_FILE" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --html_path "$HTML_PATH" \
  --generator_path "$GENERATOR_PATH" \
  --renderer_path "$RENDERER_PATH" \
  "${EXTRA_RENDER_ARGS[@]}" \
  "${LORA_ARGS[@]}" \
  --adapter_path "$ADAPTER_PATH" \
  --adapter_type unitalk_last_layer \
  --adapter_num_layers 12 \
  --adapter_dropout 0.0 \
  --adapter_window_mode lookahead \
  --adapter_future_steps 6 \
  --ref_path "$REF_PATH" \
  --wav2vec_model_path "$WAV2VEC_MODEL_PATH" \
  --moshi_root "$PERSONAPLEX_DIR" \
  --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --moshi_weight "$PERSONAPLEX_DIR/model_bnb_4bit.pt" \
  --mimi_weight "$PERSONAPLEX_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" \
  --tokenizer "$PERSONAPLEX_DIR/tokenizer_spm_32k_3.model" \
  --quantize_4bit \
  --text_prompt "$ROBERT_PROMPT" \
  --voice_prompt "$VOICE_PROMPT" \
  --voice_prompt_dir "$VOICE_PROMPT_DIR" \
  --enable_moshi_reply \
  --direct_reply_hidden \
  --reply_hidden_steps_per_chunk 12 \
  --audio_chunk_sec 0.96 \
  --wav2vec_sec 0.96 \
  --fm_chunk_frames 24 \
  --prebuffer_chunks 1 \
  --render_sub_batch 8 \
  --renderer_precision fp32 \
  --frame_q_backpressure 32 \
  --buffer_ms 160 \
  --skip_fm_audio_encoder \
  --assistant_speech_rms_threshold "${ASSISTANT_SPEECH_RMS_THRESHOLD:-0.006}" \
  --assistant_speech_hold_chunks "${ASSISTANT_SPEECH_HOLD_CHUNKS:-1}" \
  --a_cfg_scale "$A_CFG_SCALE" \
  --nfe "$NFE" \
  --seed 42 \
  --noise_seed 42 \
  --shared_noise \
  --fp32 \
  --tf32 \
  --dump_motion \
  --dump_dir "$DUMP_DIR" \
  --silence_helium_path "$SILENCE_HELIUM_PATH" \
  --jpeg_quality 58 \
  --device cuda \
  --reply_audio_gain 1.0 \
  --output_audio_codec opus
