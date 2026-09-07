#!/usr/bin/env bash
set -euo pipefail

ROOT="${SPEECH2AVATAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IMTALKER_DIR="$ROOT/IMTalker"
CHECKPOINT_DIR="$ROOT/checkpoints"
PERSONAPLEX_DIR="$CHECKPOINT_DIR/personaplex_bnb4"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
usage() {
  cat <<'EOF'
Usage: ./prepare_imtalker_personaplex.sh --hf-token TOKEN

  --hf-token TOKEN  Use TOKEN to download the required Hugging Face assets.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-token)
      [[ $# -ge 2 && -n "$2" ]] || {
        echo "--hf-token requires a token value." >&2
        usage >&2
        exit 2
      }
      HF_TOKEN="$2"
      export HF_TOKEN
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

[[ -f "$IMTALKER_DIR/requirement.txt" ]] || {
  echo "Run this script from a complete speech2avatar clone." >&2
  exit 1
}

if [[ "$(id -u)" -eq 0 ]]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.11 python3.11-venv \
    ffmpeg git git-lfs htop tmux curl ca-certificates build-essential
  git lfs install
else
  echo "Not root: skipping apt packages. Python 3.11, ffmpeg, git-lfs, and build tools must already exist."
fi

command -v "$PYTHON_BIN" >/dev/null || {
  echo "Missing $PYTHON_BIN." >&2
  exit 1
}

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel
python -m pip install "setuptools==80.9.0"
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$IMTALKER_DIR/requirement.txt"
python -m pip install \
  "huggingface_hub[cli]==0.36.2" \
  hf_transfer tensorboard \
  "sphn==0.2.1" einops sentencepiece \
  "aiohttp==3.14.3" "av==17.1.0" "aiortc==1.15.0" \
  "bitsandbytes==0.50.0"

if [[ -z "${HF_TOKEN:-}" ]] && ! hf auth whoami >/dev/null 2>&1; then
  echo "Hugging Face access is required for the gated PersonaPlex assets." >&2
  echo "Run again with: ./prepare_imtalker_personaplex.sh --hf-token TOKEN" >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p \
  "$IMTALKER_DIR/checkpoints/wav2vec2-base-960h" \
  "$CHECKPOINT_DIR/fullgen_static_2s_6400_resume" \
  "$CHECKPOINT_DIR/personaplex_unitalk_strict2s_2gpu_15k" \
  "$CHECKPOINT_DIR/lora" \
  "$CHECKPOINT_DIR/personaplex_lookahead_rms_adapter/stats" \
  "$PERSONAPLEX_DIR"

echo "[1/7] IMTalker renderer and Wav2Vec files"
for file in \
  renderer.ckpt \
  wav2vec2-base-960h/config.json \
  wav2vec2-base-960h/pytorch_model.bin \
  wav2vec2-base-960h/preprocessor_config.json \
  wav2vec2-base-960h/feature_extractor_config.json; do
  hf download cbsjtu01/IMTalker "$file" --local-dir "$IMTALKER_DIR/checkpoints"
done

echo "[2/7] Two-second IMTalker generator and adapter"
hf download niloy629/hdtf_preprocess \
  live_winner/fullgen_static_2s_6400_resume/last.ckpt \
  live_winner/adapters/personaplex_unitalk_strict2s_2gpu_15k_last.pt \
  --repo-type dataset --local-dir "$CHECKPOINT_DIR"
ln -sfn \
  "$CHECKPOINT_DIR/live_winner/fullgen_static_2s_6400_resume/last.ckpt" \
  "$CHECKPOINT_DIR/fullgen_static_2s_6400_resume/last.ckpt"
ln -sfn \
  "$CHECKPOINT_DIR/live_winner/adapters/personaplex_unitalk_strict2s_2gpu_15k_last.pt" \
  "$CHECKPOINT_DIR/personaplex_unitalk_strict2s_2gpu_15k/last.pt"

echo "[3/7] Blink motion and silence Helium seed"
hf download niloy629/hdtf_preprocess \
  lora/3robert_audio3_ditto_static_motion.pt \
  personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt \
  --repo-type dataset --local-dir "$CHECKPOINT_DIR"

echo "[4/7] PersonaPlex bnb4 package and weights"
hf download brianmatzelle/personaplex-7b-v1-bnb-4bit \
  --local-dir "$PERSONAPLEX_DIR"

echo "[5/7] PersonaPlex Mimi and tokenizer"
hf download nvidia/personaplex-7b-v1 \
  tokenizer-e351c8d8-checkpoint125.safetensors \
  tokenizer_spm_32k_3.model \
  --local-dir "$PERSONAPLEX_DIR"

echo "[6/7] PersonaPlex voices and bundled Robert_5 voice"
hf download nvidia/personaplex-7b-v1 voices.tgz --local-dir "$PERSONAPLEX_DIR"
tar --no-same-owner -xzf "$PERSONAPLEX_DIR/voices.tgz" -C "$PERSONAPLEX_DIR"
install -m 0644 "$ROOT/bundled_assets/Robert_5.pt" "$PERSONAPLEX_DIR/voices/Robert_5.pt"

echo "[7/7] Install bundled Moshi and verify deployment"
[[ -f "$PERSONAPLEX_DIR/moshi/pyproject.toml" ]] || {
  echo "PersonaPlex download is missing bundled moshi source." >&2
  exit 1
}
python -m pip install -e "$PERSONAPLEX_DIR/moshi" --no-deps

# --- Optional: STT + query routing + web search dependencies/assets --------
# Opt-in via ENABLE_SEARCH=1 (default 0): this is the ONLY thing that changes
# what gets installed/downloaded here. Skipping it reproduces the exact
# install this script has always done, with zero new packages or downloads.
ENABLE_SEARCH="${ENABLE_SEARCH:-0}"
if [[ "$ENABLE_SEARCH" == "1" ]]; then
  echo "[search 1/3] Overriding transformers and installing peft for the STT submodel and the Qwen router/compressor"
  # requirement.txt pins transformers==4.30.2 for IMTalker's own long-stable
  # Wav2Vec2FeatureExtractor usage. Qwen2.5 support and chat templates need a
  # much newer transformers, so this reinstalls it *last*, deliberately
  # overriding the pin -- IMTalker's own transformers usage is unaffected.
  # aiohttp is intentionally left at the version requirement.txt/above already
  # installed (3.14.3): the web-search code path only uses basic
  # ClientSession/post/get APIs that are stable across that range, and this
  # pipeline already depends on that newer aiohttp elsewhere.
  python -m pip install "peft>=0.19,<0.20" "transformers==4.52.4"

  STT_PKG_DIR="${STT_PKG_DIR:-$CHECKPOINT_DIR/stt}"
  echo "[search 2/3] Installing an isolated copy of the upstream Kyutai moshi package (STT submodel only)"
  # Installed with --target into its own directory, never into the venv's
  # normal site-packages: this project's own PersonaPlex fork already owns
  # the top-level `moshi` import name, and the two packages cannot coexist in
  # sys.modules under the same name (see IMTalker/search_helpers.py).
  mkdir -p "$STT_PKG_DIR"
  python -m pip install --no-deps --target "$STT_PKG_DIR" moshi

  echo "[search 3/3] Downloading the <lookup>/<ref> reference LoRA adapter"
  # Optional: only needed when the live server is launched with
  # ENABLE_SEARCH=1. This adapter teaches PersonaPlex to correctly use the
  # <lookup>/<ref> tags that carry injected web-search context. Never fails
  # the whole prepare run if the repo is unreachable -- search is additive,
  # the avatar must still be able to boot without it (with ENABLE_SEARCH=0).
  REF_LORA_DIR="${REF_LORA_DIR:-$CHECKPOINT_DIR/rag_lora}"
  if mkdir -p "$REF_LORA_DIR/lora" && hf download Darknsu/helium_lora_v1 \
    adapter_model.safetensors \
    --repo-type dataset \
    --local-dir "$REF_LORA_DIR/lora"
  then
    # adapter_config.json is not published in that dataset repo (only the
    # weights are) -- write the matching config by hand, exactly as the
    # adapter was trained/saved with.
    cat > "$REF_LORA_DIR/lora/adapter_config.json" <<'JSON'
{
  "alora_invocation_tokens": null,
  "alpha_pattern": {},
  "arrow_config": null,
  "auto_mapping": null,
  "base_model_name_or_path": null,
  "bias": "none",
  "corda_config": null,
  "ensure_weight_tying": false,
  "eva_config": null,
  "exclude_modules": null,
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layer_replication": null,
  "layers_pattern": null,
  "layers_to_transform": null,
  "loftq_config": {},
  "lora_alpha": 256.0,
  "lora_bias": false,
  "lora_dropout": 0.05,
  "lora_ga_config": null,
  "megatron_config": null,
  "megatron_core": "megatron.core",
  "modules_to_save": null,
  "peft_type": "LORA",
  "peft_version": "0.19.1",
  "qalora_group_size": 16,
  "r": 128,
  "rank_pattern": {},
  "revision": null,
  "target_modules": ["proj", "fc1", "out_proj", "fc2", "linear", "in_proj"],
  "target_parameters": null,
  "task_type": "FEATURE_EXTRACTION",
  "trainable_token_indices": null,
  "use_bdlora": null,
  "use_dora": false,
  "use_qalora": false,
  "use_rslora": false
}
JSON
    echo "  reference LoRA ready: $REF_LORA_DIR/lora"
  else
    echo "  [warn] reference LoRA download failed/unreachable -- continuing without it." >&2
    echo "         Re-run this script with ENABLE_SEARCH=1 later, or launch with ENABLE_SEARCH=0." >&2
  fi

  # Qwen/Qwen2.5-1.5B-Instruct (the shared router + compressor model) and
  # kyutai/stt-1b-en_fr-candle (the STT submodel) are NOT pre-fetched here --
  # both are ordinary (non-gated) HF Hub repos that transformers/moshi
  # download and cache on first use. Pre-warm the cache with a plain `hf
  # download Qwen/Qwen2.5-1.5B-Instruct` / `hf download kyutai/stt-1b-en_fr-candle`
  # beforehand if you want the first live-server launch to skip that download.
else
  echo "[search] ENABLE_SEARCH is not 1 -- skipping STT/routing/web-search dependencies and assets."
fi

SPEECH2AVATAR_ROOT="$ROOT" VENV_DIR="$VENV_DIR" ENABLE_SEARCH="$ENABLE_SEARCH" \
  REF_LORA_DIR="${REF_LORA_DIR:-}" STT_PKG_DIR="${STT_PKG_DIR:-}" \
  "$ROOT/run_imtalker_personaplex.sh" --check-only

echo
echo "Preparation complete. Start the server with:"
echo "  cd $ROOT && bash run_imtalker_personaplex.sh"
