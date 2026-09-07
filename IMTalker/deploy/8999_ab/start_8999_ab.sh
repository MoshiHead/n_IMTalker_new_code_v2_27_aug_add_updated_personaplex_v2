#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMTALKER_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${IMTALKER_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
PORT="${PORT:-8999}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}"
PERSONAPLEX_ROOT="${PERSONAPLEX_ROOT:-${CHECKPOINT_ROOT}/personaplex_bnb4}"
VOICE_ROOT="${VOICE_ROOT:-${CHECKPOINT_ROOT}/personaplex_voices}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${CHECKPOINT_ROOT}/personaplex_imtalker_live_8999}"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${IMTALKER_ROOT}:${PERSONAPLEX_ROOT}/moshi:${PERSONAPLEX_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export IMTALKER_PROMPT_STATE_CACHE=1

ROBERT_PROMPT="$(tr '\n' ' ' < "${IMTALKER_ROOT}/prompts/RB_Robert_System_Prompt_full.txt")"

cd "${IMTALKER_ROOT}"
exec "${PYTHON_BIN}" -u "${SCRIPT_DIR}/live_personaplex_imtalker_ab.py" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --html_path "${IMTALKER_ROOT}/static/index_v3_binary_fullscreen_ab.html" \
  --generator_path "${IMTALKER_ROOT}/checkpoints/generator.ckpt" \
  --renderer_path "${IMTALKER_ROOT}/checkpoints/renderer.ckpt" \
  --lora_generator_path "${CHECKPOINT_ROOT}/lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt" \
  --lora_rank 64 \
  --lora_alpha 128 \
  --lora_dropout 0.05 \
  --adapter_path "${ADAPTER_ROOT}/unitalk_live_turn_silence_ft_epoch_000010.pt" \
  --adapter_type unitalk_last_layer \
  --adapter_num_layers 12 \
  --adapter_dropout 0.0 \
  --adapter_window_mode lookahead \
  --adapter_future_steps 6 \
  --ref_path "${IMTALKER_ROOT}/assets/3robert.jpeg" \
  --wav2vec_model_path "${IMTALKER_ROOT}/checkpoints/wav2vec2-base-960h" \
  --moshi_root "${PERSONAPLEX_ROOT}" \
  --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --moshi_weight "${PERSONAPLEX_ROOT}/model_bnb_4bit.pt" \
  --mimi_weight "${PERSONAPLEX_ROOT}/hf_assets/tokenizer-e351c8d8-checkpoint125.safetensors" \
  --tokenizer "${PERSONAPLEX_ROOT}/hf_assets/tokenizer_spm_32k_3.model" \
  --quantize_4bit \
  --text_prompt "${ROBERT_PROMPT}" \
  --voice_prompt NATM0.pt \
  --voice_prompt_dir "${VOICE_ROOT}" \
  --enable_moshi_reply \
  --direct_reply_hidden \
  --reply_hidden_steps_per_chunk 12 \
  --audio_chunk_sec 0.96 \
  --wav2vec_sec 0.96 \
  --fm_chunk_frames 24 \
  --prebuffer_chunks 1 \
  --skip_fm_audio_encoder \
  --a_cfg_scale 1.15 \
  --nfe 5 \
  --seed 42 \
  --noise_seed 42 \
  --shared_noise \
  --fp32 \
  --tf32 \
  --dump_motion \
  --dump_dir "${IMTALKER_ROOT}/live_dumps_typeab_robert_full_8999" \
  --silence_helium_path "${ADAPTER_ROOT}/silence_helium_mean.pt" \
  --jpeg_quality 58 \
  --device cuda \
  --reply_audio_gain 1.0
