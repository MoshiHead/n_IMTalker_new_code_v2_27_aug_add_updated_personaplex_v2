#!/usr/bin/env bash
set -euo pipefail
ROOT="${SPEECH2AVATAR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IM="$ROOT/IMTalker"
BNB="$ROOT/checkpoints/personaplex_bnb4"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"
PORT="${PORT:-8998}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
VOICE_PROMPT="${VOICE_PROMPT:-Robert_5.pt}"
DEFAULT_PROMPT_FILE="$IM/prompts/Robert_8998_default.txt"
TEXT_PROMPT_FILE="${TEXT_PROMPT_FILE:-$DEFAULT_PROMPT_FILE}"
PROMPT_CACHE="${PROMPT_CACHE:-0}"
CHECK_ONLY=0
[[ "${1:-}" == "--check-only" ]] && CHECK_ONLY=1
if [[ -n "${TEXT_PROMPT:-}" ]]; then
  TEXT_PROMPT_VALUE="$TEXT_PROMPT"
elif [[ -f "$TEXT_PROMPT_FILE" ]]; then
  TEXT_PROMPT_VALUE="$(<"$TEXT_PROMPT_FILE")"
else
  echo "Missing text prompt file: $TEXT_PROMPT_FILE" >&2
  exit 1
fi
case "$PROMPT_CACHE" in
  0|false|no|off) PROMPT_CACHE=0 ;;
  1|true|yes|on) PROMPT_CACHE=1 ;;
  *) echo "PROMPT_CACHE must be 0 or 1." >&2; exit 2 ;;
esac

# STT + query routing + web search -- opt-in, additive. ENABLE_SEARCH=0
# (default) reproduces the plain conversational launch command with zero new
# flags appended. Web search itself is a further opt-in on top of that: the
# router still runs with ENABLE_SEARCH=1 alone, but a "needs search" turn
# falls back to the model's own knowledge unless WEB_SEARCH_ENABLED=1 and a
# WEB_SEARCH_API_KEY are also set.
ENABLE_SEARCH="${ENABLE_SEARCH:-0}"
if [[ "$ENABLE_SEARCH" == "1" ]]; then
  REF_LORA_DIR="${REF_LORA_DIR:-$ROOT/checkpoints/rag_lora}"
  STT_PKG_DIR="${STT_PKG_DIR:-$ROOT/checkpoints/stt}"
  CONVERSATION_LOG_DIR="${CONVERSATION_LOG_DIR:-$ROOT/conversation_logs}"
  THINKING_SOUND_PATH="${THINKING_SOUND_PATH:-$ROOT/bundled_assets/ai-thinking-sound.wav}"
  SEARCH_ARGS=(
    --conversation_log_dir "$CONVERSATION_LOG_DIR"
    --ref_lora_dir "$REF_LORA_DIR"
    --stt_hf_repo "${STT_HF_REPO:-kyutai/stt-1b-en_fr-candle}"
    --stt_pkg_dir "$STT_PKG_DIR"
    --vad_threshold "${VAD_THRESHOLD:-0.5}"
    # The bundled STT model is English/French only, so a transcript in another
    # script is decode garbage, not a language surprise. Set
    # STT_REJECT_FOREIGN_SCRIPT=0 only for a deliberately multilingual STT
    # checkpoint.
    --stt_reject_foreign_script "${STT_REJECT_FOREIGN_SCRIPT:-1}"
    --stt_max_non_latin_ratio "${STT_MAX_NON_LATIN_RATIO:-0.15}"
    # Also drop Latin-script transcripts that are not English (the STT model
    # is bilingual en/fr and hallucinates Spanish/French on unclear audio).
    --stt_require_english "${STT_REQUIRE_ENGLISH:-1}"
    # Hold the model silent for the whole search instead of only muting its
    # audio -- muting alone lets it compose an invented figure behind the
    # filler and finish that sentence even after the real <ref> arrived.
    --suppress_text_during_search "${SUPPRESS_TEXT_DURING_SEARCH:-1}"
    # One small instruct model does double duty: it routes every transcript
    # (search / no search) AND compresses web results into one spoken
    # sentence. Sharing it means routing costs no extra VRAM and no extra
    # load time. Omitting this flag disables routing and search entirely.
    --compressor_model "${COMPRESSOR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
    --compressor_device "${COMPRESSOR_DEVICE:-cuda}"
    --compressor_4bit "${COMPRESSOR_4BIT:-1}"
    --compressor_max_passages "${COMPRESSOR_MAX_PASSAGES:-2}"
    # Router bias. Below 0.5 on purpose: an unnecessary search costs ~2s of
    # thinking sound and is recoverable, while a missed search produces a
    # confidently wrong answer spoken aloud.
    --router_threshold "${ROUTER_THRESHOLD:-0.40}"
    # 1 = run the instant regex pre-pass first, so obvious cases never pay
    # for a model forward pass. 0 = route every turn through the model.
    --router_rules "${ROUTER_RULES:-1}"
    --thinking_sound_path "$THINKING_SOUND_PATH"
    # Real search+compression latency runs ~2.5-3.7s end to end; the fallback
    # cap must sit comfortably above that or a correctly-computed answer gets
    # discarded before it lands.
    --search_max_filler_sec "${SEARCH_MAX_FILLER_SEC:-6.0}"
    # Web results have no relevance floor otherwise -- search engines always
    # return something, so this is what stands between an unrelated page and
    # the assistant's spoken answer.
    --web_search_min_score "${WEB_SEARCH_MIN_SCORE:-0.15}"
    --max_ref_tokens "${MAX_REF_TOKENS:-250}"
    # Forensic fix (RunPod RTX 5090 run, 2026-09-05): the LLM compressor was
    # measured taking 2.0-5.0s per call -- by far the largest latency source,
    # and what raced (and lost to) the filler timeout above, discarding a
    # correctly-computed answer. extractive_first tries a free, ~0ms
    # best-sentence extraction before ever paying for that LLM forward pass.
    --compressor_mode "${COMPRESSOR_MODE:-extractive_first}"
    --extractive_confidence_threshold "${EXTRACTIVE_CONFIDENCE_THRESHOLD:-0.55}"
    # Independent of the filler timeout: hard cap on how long the model is
    # held forcibly silent waiting on a slow search/compress. The search
    # keeps running past this point; only the forced silence is released.
    --max_suppress_sec "${MAX_SUPPRESS_SEC:-3.0}"
    # Spread a <ref> injection across several 80ms ticks instead of blocking
    # the real-time GPU thread in one call -- keeps mic ingestion and avatar
    # rendering close to their normal cadence during injection.
    #
    # REVERTED 14 -> 4 (logs_5). Raising this to 14 to save ~1.2s of search
    # latency MUTED every search answer. Each forced token is one extra
    # lm_gen._step(), i.e. one extra 12.5Hz frame of model time, so 14/tick
    # advances PersonaPlex ~15x faster than the wall clock it is streaming
    # against and desynchronises its text codebook from its audio codebooks.
    # The A/B is unambiguous -- same code, only this value differing:
    #   logs_4 @4/tick  search turns delivered 79 / 72 / 97 audio packets
    #   logs_5 @14/tick search turns delivered  8 /  8 /  5 audio packets
    # and logs_5 turn 4 proves it is an AUDIO failure, not a text one: 88
    # chars of correct answer text ("...price today is 309.32, up 0.23%...")
    # came out with 0.64s of audible audio behind it. Do not raise this
    # without re-testing search audio on real hardware.
    --inject_tokens_per_tick "${INJECT_TOKENS_PER_TICK:-4}"
    # If no real text/audio follows an injection within this many seconds,
    # log it immediately instead of waiting for the next question's VAD to
    # notice minutes later (observed: turn 7 in the same run went silent for
    # 50+ seconds after a clean, on-time injection).
    --post_inject_watchdog_sec "${POST_INJECT_WATCHDOG_SEC:-4.0}"
    # Forensic fix (logs_2, RunPod RTX 5090 run 2026-09-06): the assistant was
    # SPEAKING THE INJECTED REFERENCE ALOUD before its real answer -- turn 6
    # audibly produced ". Class ( stock $. reflecting. move opened>" and turn 5
    # ". As02, for $1ref> ...". The injected steps themselves are silent, but
    # PersonaPlex's audio codebooks lag its text stream, so the acoustic
    # rendition of those forced tokens arrives over the following steps. This
    # keeps the outgoing audio covered (thinking sound, or silence if no clip
    # is set) for that lag, so the user hears the waiting cue and then the real
    # answer -- never the raw reference. Set 0 to disable the mask.
    --ref_audio_drain_sec "${REF_AUDIO_DRAIN_SEC:-2.0}"
  )
  # Web search is what "needs live data" resolves to, so default it ON
  # whenever a key is available. Without a key the router still runs and
  # still decides -- turns that need live data just fall back to the model's
  # own knowledge instead of hanging.
  if [[ -n "${WEB_SEARCH_API_KEY:-}" ]]; then
    WEB_SEARCH_ENABLED="${WEB_SEARCH_ENABLED:-1}"
  else
    WEB_SEARCH_ENABLED="${WEB_SEARCH_ENABLED:-0}"
  fi
  if [[ "$WEB_SEARCH_ENABLED" == "1" ]]; then
    SEARCH_ARGS+=(
      --web_search_enabled
      --web_search_api_key "${WEB_SEARCH_API_KEY:?set WEB_SEARCH_API_KEY when WEB_SEARCH_ENABLED=1}"
      --web_search_provider "${WEB_SEARCH_PROVIDER:-tavily}"
      --web_search_max_results "${WEB_SEARCH_MAX_RESULTS:-3}"
      --web_search_timeout "${WEB_SEARCH_TIMEOUT:-3.0}"
    )
  else
    echo "[warn] ENABLE_SEARCH=1 but no WEB_SEARCH_API_KEY -- the router will still run," >&2
    echo "       but questions needing live data will fall back to the model's own knowledge." >&2
  fi
  echo "Search enabled: ref_lora=$REF_LORA_DIR stt_pkg=$STT_PKG_DIR web_search=$WEB_SEARCH_ENABLED provider=${WEB_SEARCH_PROVIDER:-tavily} router_threshold=${ROUTER_THRESHOLD:-0.40} conversation_log_dir=$CONVERSATION_LOG_DIR"
else
  SEARCH_ARGS=()
fi

required=(
 "$VENV_DIR/bin/python"
 "$IM/imtalker_personaplex_try_vad2_8998.py" "$IM/seedvc_runtime.py" "$IM/liveTry.py" "$IM/liveTry_cached.py" "$IM/ws_av_binary_codec.py"
 # runtime_logging.py and conversation_logger.py are imported unconditionally
 # by liveTry.py/liveTry_cached.py regardless of ENABLE_SEARCH -- a missing
 # file here is an ImportError at engine construction, not a degraded mode.
 "$IM/runtime_logging.py" "$IM/conversation_logger.py" "$IM/latency_logger.py"
 "$IM/experiments/original_pod_8998/FM.py" "$IM/experiments/original_pod_8998/FMT.py"
 "$IM/static/index_v3_binary_fullscreen_robot_try_vad2.html" "$IM/static/assets/robert_idle_10s.mp4" "$IM/static/assets/audio-processor-aj-nodrop.js"
 "$IM/static/assets/decoderWorker.min.js" "$IM/static/assets/decoderWorker.min.wasm" "$IM/static/assets/encoderWorker.min-DpsJ02BN.js"
 "$IM/assets/3robert.jpeg" "$IM/checkpoints/renderer.ckpt" "$IM/checkpoints/wav2vec2-base-960h/config.json"
 "$ROOT/checkpoints/fullgen_static_2s_6400_resume/last.ckpt" "$ROOT/checkpoints/personaplex_unitalk_strict2s_2gpu_15k/last.pt"
 "$ROOT/checkpoints/lora/3robert_audio3_ditto_static_motion.pt"
 "$ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt"
 "$BNB/model_bnb_4bit.pt" "$BNB/tokenizer-e351c8d8-checkpoint125.safetensors" "$BNB/tokenizer_spm_32k_3.model" "$BNB/voices/$VOICE_PROMPT"
)
for path in "${required[@]}"; do [[ -e "$path" ]] || { echo "Missing required file: $path" >&2; exit 1; }; done
if [[ "$ENABLE_SEARCH" == "1" ]]; then
  [[ -e "$REF_LORA_DIR/lora/adapter_config.json" ]] || {
    echo "Missing required search path: $REF_LORA_DIR/lora/adapter_config.json (re-run prepare_imtalker_personaplex.sh with ENABLE_SEARCH=1, or set ENABLE_SEARCH=0)" >&2
    exit 1
  }
  [[ -e "$IM/search_helpers.py" ]] || { echo "Missing required search path: $IM/search_helpers.py" >&2; exit 1; }
  # Loaded and cached ONCE at engine init (never re-read per turn). Warn rather
  # than exit: a missing clip degrades to forced silence during the wait, which
  # is worse UX but not a broken pipeline.
  [[ -e "$THINKING_SOUND_PATH" ]] || {
    echo "[warn] thinking sound not found at $THINKING_SOUND_PATH -- searches will wait in silence" >&2
  }
fi
declare -A hashes=(
 ["$IM/imtalker_personaplex_try_vad2_8998.py"]="23424d06480e29593cd721838e27c1ddeaec7e5ab772c50c3c92f5f615398f1f"
 ["$IM/static/index_v3_binary_fullscreen_robot_try_vad2.html"]="5cf3981351668e0366b7b4adf2f36c7e43f5ab0c672f6616a343a72817582fa6"
 ["$IM/static/assets/robert_idle_10s.mp4"]="6bdfb847fb3dd2a76d42278a138e26e2729bf5ed938f6733a3b428768a9e7916"
 ["$IM/experiments/original_pod_8998/FM.py"]="8620d6cad2b945276a792a1d63159369654cbb83f9114ab5788f93a3d8daf5d9"
 ["$IM/experiments/original_pod_8998/FMT.py"]="286eb512e710926b0a88d1bc47f14aef5cfc3ef6fc0987fc3cf0d9e7bd004c5d"
 ["$IM/liveTry.py"]="201ee853c21e6b0b9b4dc590928734e2236ef11c69d1143afa80257333b46935"
 ["$IM/liveTry_cached.py"]="32e6818c7f7e138323e9eabe7f21cca365a4828893154af043b9c040f89dbf2e"
 ["$IM/seedvc_runtime.py"]="fe46773af65e010e3d6f41732f0fa1c3e3cf6a8221d9c68718e15561062337f7"
 ["$IM/ws_av_binary_codec.py"]="c090b6a5a076743055f1dd34301662405a28d5cb1636556e9de4c895ddffe4d3"
 ["$BNB/voices/Robert_5.pt"]="a9684503d2a9d37f527341c9a0385b9ed0943eac955b40159bc34f4796563c3d"
 ["$ROOT/checkpoints/fullgen_static_2s_6400_resume/last.ckpt"]="000d595124516f6437e218213a31c2ede2350ebfda7bb121a957ef5d52b0e88e"
 ["$ROOT/checkpoints/personaplex_unitalk_strict2s_2gpu_15k/last.pt"]="c9c86d108f81fbdef57e1548ca403b78a68acc32c5a37dab12265d72654f55b9"
 ["$ROOT/checkpoints/lora/3robert_audio3_ditto_static_motion.pt"]="e29a41ff004b228d7efee15cad0f32f4d4bc5466563709e2ba78b158d4e340bb"
 ["$IM/checkpoints/renderer.ckpt"]="ca1686c1157b8ef5de43eabdeb846db4612694f5f74012be38742b0871808755"
 ["$ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt"]="20a6d6eb58608d6d202bac46958e595e243635fdeeb8f04eb1afbe2ac7f2f16d"
)
for path in "${!hashes[@]}"; do actual="$(sha256sum "$path" | awk '{print $1}')"; [[ "$actual" == "${hashes[$path]}" ]] || { echo "Checksum mismatch: $path" >&2; exit 1; }; done
source "$VENV_DIR/bin/activate"
python - <<'PY'
import torch, torchaudio, bitsandbytes, aiohttp, av, sphn
assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.__version__.startswith("2.8.0+cu128"), torch.__version__
print("CUDA:", torch.cuda.get_device_name(0)); print("Torch:", torch.__version__)
PY
python -m py_compile "$IM/imtalker_personaplex_try_vad2_8998.py" "$IM/seedvc_runtime.py" "$IM/liveTry.py" "$IM/liveTry_cached.py" "$IM/experiments/original_pod_8998/FM.py" "$IM/experiments/original_pod_8998/FMT.py" "$IM/runtime_logging.py" "$IM/conversation_logger.py" "$IM/latency_logger.py" "$IM/search_helpers.py"
echo "Preflight OK: try_vad2, $VOICE_PROMPT, prompt cache=$PROMPT_CACHE, 2.0s/25-step chunks, 50 frames, CFG 1.24, NFE 3, renderer sub-batch 6, FP32, Opus. search=$ENABLE_SEARCH web_search=${WEB_SEARCH_ENABLED:-0}"
[[ "$CHECK_ONLY" -eq 1 ]] && exit 0
if ss -ltnp | grep -q ":${PORT}\\b"; then echo "Port $PORT is occupied:" >&2; ss -ltnp | grep ":${PORT}\\b" >&2; exit 1; fi
# LOGS_DIR: where runtime_logging.py (system_runtime.log, conversation.log)
# writes, independent of $IM/logs above (which is unrelated, pre-existing
# scratch space). Exported explicitly so it resolves correctly regardless of
# whether SPEECH2AVATAR_ROOT happens to be set in the caller's environment.
export LOGS_DIR="${LOGS_DIR:-$ROOT/logs}"
export SPEECH2AVATAR_ROOT="$ROOT"
mkdir -p "$IM/logs" "$LOGS_DIR" "$ROOT/pacing_compare/integrated"
cd "$IM"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$IM:$BNB/moshi:$BNB:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
export IMTALKER_CACHED_ENGINE="$PROMPT_CACHE" IMTALKER_PROMPT_STATE_CACHE="$PROMPT_CACHE" IMTALKER_TRANSITION_BLEND_FRAMES=0
exec python -u "$IM/imtalker_personaplex_try_vad2_8998.py" \
 --host 0.0.0.0 --port "$PORT" --html_path "$IM/static/index_v3_binary_fullscreen_robot_try_vad2.html" \
 --generator_path "$ROOT/checkpoints/fullgen_static_2s_6400_resume/last.ckpt" --renderer_path "$IM/checkpoints/renderer.ckpt" \
 --adapter_path "$ROOT/checkpoints/personaplex_unitalk_strict2s_2gpu_15k/last.pt" \
 --adapter_type unitalk_last_layer --adapter_num_layers 12 --adapter_dropout 0.0 --adapter_window_mode lookahead --adapter_future_steps 0 \
 --ref_path "$IM/assets/3robert.jpeg" --wav2vec_model_path "$IM/checkpoints/wav2vec2-base-960h" \
 --moshi_root "$BNB" --mimi_hf_repo nvidia/personaplex-7b-v1 --moshi_weight "$BNB/model_bnb_4bit.pt" \
 --mimi_weight "$BNB/tokenizer-e351c8d8-checkpoint125.safetensors" --tokenizer "$BNB/tokenizer_spm_32k_3.model" \
 --text_prompt "$TEXT_PROMPT_VALUE" \
 --quantize_4bit --voice_prompt "$VOICE_PROMPT" --voice_prompt_dir "$BNB/voices" \
 --enable_moshi_reply --direct_reply_hidden --reply_hidden_steps_per_chunk 25 \
 --audio_chunk_sec 2.0 --wav2vec_sec 2.0 --fm_chunk_frames 50 --helium_deque_size 25 \
 --prebuffer_chunks 1 --render_sub_batch 6 --renderer_precision fp32 --frame_q_backpressure 32 --buffer_ms 160 --skip_fm_audio_encoder \
 --assistant_speech_rms_threshold 0.006 --assistant_speech_hold_chunks 1 --motion_ref_blend 0.0 --motion_prior_noise_blend 0.0 \
 --a_cfg_scale 1.24 --nfe 3 --seed 42 --noise_seed 42 --shared_noise --fp32 --tf32 \
 --silence_helium_path "$ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt" \
 --jpeg_quality 90 --device cuda --reply_audio_gain 1.0 --output_audio_codec opus \
 --blink_motion_path "$ROOT/checkpoints/lora/3robert_audio3_ditto_static_motion.pt" --enable_eye_blink_composite \
 --suppress_media_watchdog_sec "${SUPPRESS_MEDIA_WATCHDOG_SEC:-3.0}" \
 --max_event_backlog_sec "${MAX_EVENT_BACKLOG_SEC:-0.6}" \
 "${SEARCH_ARGS[@]}"
