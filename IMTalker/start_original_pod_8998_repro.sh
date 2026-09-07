#!/usr/bin/env bash
set -euo pipefail

IMTALKER_DIR="${IMTALKER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$IMTALKER_DIR")}"
VENV_DIR="${VENV_DIR:-/workspace/preprocess_5090}"
PORT="${PORT:-8998}"

GENERATOR_PATH="${GENERATOR_PATH:-$PROJECT_ROOT/checkpoints/fullgen_static_2s_6400_resume/last.ckpt}"
ADAPTER_PATH="${ADAPTER_PATH:-$PROJECT_ROOT/checkpoints/personaplex_unitalk_strict2s_2gpu_15k/last.pt}"
BLINK_MOTION_PATH="${BLINK_MOTION_PATH:-$PROJECT_ROOT/checkpoints/lora/3robert_audio3_ditto_static_motion.pt}"
PERSONAPLEX_DIR="${PERSONAPLEX_DIR:-$PROJECT_ROOT/checkpoints/personaplex_bnb4}"
SILENCE_HELIUM_PATH="${SILENCE_HELIUM_PATH:-$PROJECT_ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt}"
VOICE_PROMPT_DIR="${VOICE_PROMPT_DIR:-$PERSONAPLEX_DIR/voices}"

for required in \
  "$GENERATOR_PATH" "$ADAPTER_PATH" "$BLINK_MOTION_PATH" \
  "$IMTALKER_DIR/checkpoints/renderer.ckpt" \
  "$IMTALKER_DIR/assets/3robert.jpeg" \
  "$IMTALKER_DIR/prompts/RB_Robert_System_Prompt_full.txt" \
  "$PERSONAPLEX_DIR/model_bnb_4bit.pt" \
  "$PERSONAPLEX_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" \
  "$PERSONAPLEX_DIR/tokenizer_spm_32k_3.model" \
  "$VOICE_PROMPT_DIR/VARM3.pt" \
  "$SILENCE_HELIUM_PATH"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

source "$VENV_DIR/bin/activate"
cd "$IMTALKER_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$IMTALKER_DIR:$PERSONAPLEX_DIR/moshi:$PERSONAPLEX_DIR:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export IMTALKER_CACHED_ENGINE=1
export IMTALKER_PROMPT_STATE_CACHE=1
export IMTALKER_TRANSITION_BLEND_FRAMES=0

PROMPT="$(tr '\n' ' ' < "$IMTALKER_DIR/prompts/RB_Robert_System_Prompt_full.txt")"

exec python -u "$IMTALKER_DIR/OriginalPod8998TransitionBlend.py" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --html_path "$IMTALKER_DIR/static/index_v3_binary_fullscreen_aj_nodrop.html" \
  --generator_path "$GENERATOR_PATH" \
  --renderer_path "$IMTALKER_DIR/checkpoints/renderer.ckpt" \
  --adapter_path "$ADAPTER_PATH" \
  --adapter_type unitalk_last_layer \
  --adapter_num_layers 12 \
  --adapter_dropout 0.0 \
  --adapter_window_mode lookahead \
  --adapter_future_steps 0 \
  --ref_path "$IMTALKER_DIR/assets/3robert.jpeg" \
  --wav2vec_model_path "$IMTALKER_DIR/checkpoints/wav2vec2-base-960h" \
  --moshi_root "$PERSONAPLEX_DIR" \
  --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --moshi_weight "$PERSONAPLEX_DIR/model_bnb_4bit.pt" \
  --mimi_weight "$PERSONAPLEX_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" \
  --tokenizer "$PERSONAPLEX_DIR/tokenizer_spm_32k_3.model" \
  --quantize_4bit \
  --text_prompt "$PROMPT" \
  --voice_prompt VARM3.pt \
  --voice_prompt_dir "$VOICE_PROMPT_DIR" \
  --enable_moshi_reply \
  --direct_reply_hidden \
  --reply_hidden_steps_per_chunk 25 \
  --audio_chunk_sec 2.0 \
  --wav2vec_sec 2.0 \
  --fm_chunk_frames 50 \
  --helium_deque_size 25 \
  --prebuffer_chunks 1 \
  --render_sub_batch 10 \
  --renderer_precision fp32 \
  --frame_q_backpressure 32 \
  --buffer_ms 160 \
  --skip_fm_audio_encoder \
  --assistant_speech_rms_threshold 0.006 \
  --assistant_speech_hold_chunks 1 \
  --motion_ref_blend 0.0 \
  --motion_prior_noise_blend 0.0 \
  --a_cfg_scale 1.24 \
  --nfe 3 \
  --seed 42 \
  --noise_seed 42 \
  --shared_noise \
  --fp32 \
  --tf32 \
  --dump_motion \
  --dump_dir "$IMTALKER_DIR/live_dumps_original_pod_8998_repro" \
  --silence_helium_path "$SILENCE_HELIUM_PATH" \
  --jpeg_quality 58 \
  --device cuda \
  --reply_audio_gain 1.0 \
  --output_audio_codec opus \
  --compile_renderer \
  --blink_motion_path "$BLINK_MOTION_PATH" \
  --enable_eye_blink_composite
