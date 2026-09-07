# PersonaPlex + IMTalker 8999 AB

This directory packages the exact Type AB server that ran on port 8999.

## Pipeline

```text
browser Opus microphone
-> 24 kHz PCM
-> PersonaPlex/Mimi at 12.5 Hz
-> reply hidden [T, 4096]
-> UniTalk last-layer live adapter
-> Wav2Vec2 features
-> IMTalker flow-matching generator
-> IMTalker renderer
-> binary JPEG + 24 kHz PCM WebSocket stream
```

The server uses:

- Full Robert prompt: 5,619 characters / 1,225 PersonaPlex tokens.
- Voice: `NATM0.pt`.
- PersonaPlex CFG: `1.0`.
- IMTalker audio CFG: `1.15`.
- NFE: `5`.
- Twelve 80 ms PersonaPlex states per 0.96-second IMTalker chunk.
- Post-warmup native PersonaPlex LM/KV prompt-state caching.
- Original AB hard RMS output gate and queue behavior.

## Runtime Artifacts

Files already expected in the checkout:

```text
IMTalker/checkpoints/generator.ckpt
IMTalker/checkpoints/renderer.ckpt
IMTalker/checkpoints/wav2vec2-base-960h/
IMTalker/assets/3robert.jpeg
checkpoints/personaplex_bnb4/model_bnb_4bit.pt
checkpoints/personaplex_bnb4/hf_assets/tokenizer-e351c8d8-checkpoint125.safetensors
checkpoints/personaplex_bnb4/hf_assets/tokenizer_spm_32k_3.model
```

Downloaded by `restore_hf_artifacts.py`:

```text
niloy629/hdtf_preprocess:
  personaplex_imtalker_live_8999/checkpoints/
    unitalk_live_turn_silence_ft_epoch_000010.pt
  personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt
  lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt

nvidia/personaplex-7b-v1:
  voices.tgz -> NATM0.pt
```

The exact adapter checksum is:

```text
3394d7dd2e8d0a7f9f0aa86c45b23fc1a089ea3bf448a6515ec6c4e7599136b3
```

## Restore

From the repository root:

```bash
PYTHON_BIN=/home/ubuntu/miniforge3/envs/speech2avatar/bin/python
"${PYTHON_BIN}" IMTalker/deploy/8999_ab/restore_hf_artifacts.py \
  --project-root "$(pwd)"
```

## Launch

```bash
mkdir -p IMTalker/logs
tmux new-session -d -s personaplex_imtalker_8999 \
  "env PYTHON_BIN=/home/ubuntu/miniforge3/envs/speech2avatar/bin/python \
  CUDA_VISIBLE_DEVICES=0 PORT=8999 \
  bash IMTalker/deploy/8999_ab/start_8999_ab.sh \
  >> IMTalker/logs/personaplex_imtalker_8999.log 2>&1"
```

Startup processes the full prompt once before caching it. Follow progress with:

```bash
tail -f IMTalker/logs/personaplex_imtalker_8999.log
```

Verify locally:

```bash
ss -ltnp | grep ':8999'
curl -sS -o /tmp/personaplex_imtalker.html \
  -w '%{http_code} %{size_download}\n' \
  http://127.0.0.1:8999/
```

## Session Reset

Normal Stop/Start closes and recreates the WebSocket. Mimi, PersonaPlex
conversation KV state, text buffers, Helium history, IMTalker motion history,
and per-session queues are reset. Only model weights, CUDA graphs, voice
conditioning, and the system-prompt baseline cache persist.
