"""PersonaPlex + IMTalker AJ server with split audio/video WebSockets.

Raw PersonaPlex audio is never stored in or cleared with the video queue.
Assistant Opus audio stays on /ws/conversation, while JPEG video frames are
sent through /ws/video?session_id=... to avoid websocket head-of-line blocking.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import contextlib
import json
import logging
import os
import queue
import re

import ws_av_binary_codec as _wsbin
import sys
import threading
import time
import traceback
import types
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sphn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as T
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from transformers import Wav2Vec2FeatureExtractor

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import runtime_logging
from experiments.original_pod_8998.FM import FMGenerator
from generator.train_lora import apply_lora_to_model
from generator.helium_w2v_frontend_adapter import HeliumToWav2VecFrontendAdapter
from generator.unitalk_wav2vec_adapter import UniTalkLastLayerLiveAdapter
from generator.options.base_options import BaseOptions
from generator.wav2vec2 import Wav2VecModel
if os.environ.get("IMTALKER_CACHED_ENGINE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    from liveTry_cached import MoshiOnlyEngine
else:
    from liveTry import MoshiOnlyEngine
from renderer.models import IMTRenderer
from seedvc_runtime import SeedVCStreamingConverter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 24_000          # Mimi sample rate (24kHz)
VIDEO_FPS = 25              # IMTalker frame rate
MIMI_FRAME_SIZE = 1_920     # samples per Mimi frame (80ms @ 24kHz)
MAIN_CODEBOOKS = 8          # codebooks used for Helium input embeddings
PREBUFFER_CHUNKS = 0        # produce this many chunks before sender starts pacing
WAV2VEC_SR = 16_000
TRANSITION_BLEND_FRAMES = max(
    0, int(os.environ.get("IMTALKER_TRANSITION_BLEND_FRAMES", "5"))
)
print(
    f"[TRANSITION-BLEND] configured frames={TRANSITION_BLEND_FRAMES}",
    flush=True,
)


class PlasticityProjectionHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_ln = nn.LayerNorm(4096)
        self.net = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768),
            nn.LayerNorm(768),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_ln(x))


class PlasticityUpsampler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(768, 768, kernel_size=4, stride=4)

    def forward(self, low: torch.Tensor, target_len: int) -> torch.Tensor:
        y = self.up(low.transpose(1, 2).contiguous())
        if y.shape[-1] != int(target_len):
            y = F.interpolate(y, size=int(target_len), mode="linear", align_corners=False)
        return y.transpose(1, 2).contiguous()


class PlasticityCausalBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=12,
            dropout=0.15,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(0.1)
        self.norm2 = nn.LayerNorm(768)
        self.ff = nn.Sequential(
            nn.Linear(768, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 768),
        )
        self.drop2 = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + self.drop1(attn)
        h = self.norm2(x)
        x = x + self.drop2(self.ff(h))
        return x


class PlasticityCausalTransformer(nn.Module):
    def __init__(self, max_len: int = 2048) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([PlasticityCausalBlock() for _ in range(8)])
        self.norm = nn.LayerNorm(768)
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.causal_mask[: x.shape[1], : x.shape[1]].to(device=x.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.norm(x)


class StudioNativeLiveAdapter(nn.Module):
    """Frontend fp32 adapter live wrapper.

    Training contract:
      raw 12.5Hz Helium -> Wav2Vec2 projected frontend [T50, 768]
      live contract:
      projected frontend -> frozen Wav2Vec2 encoder -> final hidden -> IMTalker audio_projection.
    """

    def __init__(self, wav2vec_model_path: str, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.model = HeliumToWav2VecFrontendAdapter(num_layers=int(num_layers), dropout=float(dropout))
        self.wav2vec = Wav2VecModel.from_pretrained(wav2vec_model_path, local_files_only=True).eval().float()
        for param in self.wav2vec.parameters():
            param.requires_grad_(False)

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        return self.model.load_state_dict(state_dict, strict=strict)

    @torch.no_grad()
    def forward_single(self, source: torch.Tensor, target_len: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src = source.unsqueeze(0).contiguous()
        target_len = int(target_len)
        frontend_len = max(1, target_len * 2)
        frontend50 = self.model(src.float(), target_len=frontend_len).float()
        final50 = self.wav2vec.encode_from_projected_frontend(frontend50).last_hidden_state.float()
        final25 = F.interpolate(
            final50.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)[0].float().contiguous()
        return frontend50[0].float().contiguous(), final50[0].float().contiguous(), final25


class MoshiOnlyEngineWithHidden(MoshiOnlyEngine):
    """Moshi reply engine that also returns the main LM hidden for each generated step.

    Layer[-2] is exposed as a native LMGen output so Moshi can keep CUDA graph
    replay enabled. We do not use Python forward hooks in this path.
    """

    def __init__(
        self,
        *args,
        capture_layer: int = -2,
        thinking_sound_path: str = "",
        search_max_filler_sec: float = 6.0,
        compressor_mode: str = "extractive_first",
        extractive_confidence_threshold: float = 0.55,
        max_suppress_sec: float = 3.0,
        inject_tokens_per_tick: int = 4,
        post_inject_watchdog_sec: float = 4.0,
        ref_audio_drain_sec: float = 2.0,
        spell_ref_numbers: bool = False,
        **kwargs,
    ) -> None:
        self.tf_capture_layer = int(capture_layer)
        super().__init__(*args, **kwargs)

        # Forensic finding (RunPod RTX 5090 run, logs_1/conversation.log,
        # 2026-09-05): real search+compression latency was 3.4-6.6s end to
        # end, dominated by the LLM compressor (2.0-5.0s for 14-35 tokens --
        # 10-30x slower than a 1.5B model should take in isolation, most
        # likely GPU contention with the continuously-running avatar
        # pipeline). The filler cap must stay comfortably above whatever the
        # (now-optional) LLM compression path can take, so the fallback never
        # races and discards a correctly-computed answer the way it did in
        # that log's turn 4.
        self._SEARCH_MAX_FILLER_FRAMES = max(1, round(float(search_max_filler_sec) * TARGET_SR / MIMI_FRAME_SIZE))

        # -- Compression strategy: "extractive_first" (default) tries a free,
        # ~0ms, CPU-only best-sentence extraction (search_helpers.
        # extract_best_sentence) before ever paying for an LLM forward pass;
        # the LLM compressor only runs when extraction is not confident. This
        # is what cuts the 2-5s compression latency out of the common case.
        # "llm_only" reproduces the old (pre-fix) behavior; "extractive_only"
        # never calls the LLM at all. See _route_and_search.
        self.compressor_mode = str(compressor_mode or "extractive_first")
        self.extractive_confidence_threshold = float(extractive_confidence_threshold)

        # Forensic finding: turn 4's 6.6s of forced silence (suppress_text_
        # until_ref) preceded a stuck-silence failure; independent of THAT
        # bug's exact cause, holding the model artificially silent for many
        # seconds is itself undesirable. This caps how long a turn may be
        # held silent waiting on a slow search/compress, regardless of
        # whether the <ref> is ready yet -- when it expires, suppression is
        # lifted early (the model may say something generic) but the
        # in-flight search keeps running and its <ref> still gets injected
        # normally once ready.
        self.max_suppress_sec = float(max_suppress_sec)

        # Injecting N tokens synchronously in one _step() call was measured
        # blocking the real-time GPU thread for up to 1.3s at once (logs_1,
        # turn 7's ref_inject stage) -- during which no mic audio is consumed
        # and no avatar frame is produced. Spreading injection across
        # multiple 80ms ticks (a few tokens per tick) keeps each _step() call
        # close to its normal budget.
        self.inject_tokens_per_tick = max(1, int(inject_tokens_per_tick))
        # Guard rail (logs_5): every forced token is an extra lm_gen._step(),
        # i.e. an extra 12.5Hz frame of model time. Push too many into one
        # 80ms tick and PersonaPlex's audio codebooks fall out of step with
        # its text codebook -- it keeps emitting correct TEXT while its audio
        # goes silent, which is exactly how logs_5 muted all three search
        # answers at 14/tick (logs_4 at 4/tick delivered 79/72/97 packets on
        # the same questions). Loud warning rather than a hard clamp, so the
        # value stays tunable for anyone re-testing on real hardware.
        if self.inject_tokens_per_tick > self._INJECT_TOKENS_PER_TICK_SAFE_MAX:
            msg = (
                f"inject_tokens_per_tick={self.inject_tokens_per_tick} exceeds the "
                f"empirically safe maximum of {self._INJECT_TOKENS_PER_TICK_SAFE_MAX}. "
                f"On RunPod RTX 5090 this desynchronised PersonaPlex's text and audio "
                f"codebooks and muted every web-search answer (logs_5). Re-verify "
                f"search AUDIO -- not just answer text -- before trusting this value."
            )
            print(f"[liveTryPlasticity][search] WARNING {msg}", flush=True)
            try:
                runtime_logging.log_event(
                    runtime_logging.get_system_logger(), "Search", "inject_rate_above_safe_max",
                    inject_tokens_per_tick=self.inject_tokens_per_tick,
                    safe_max=self._INJECT_TOKENS_PER_TICK_SAFE_MAX,
                )
            except Exception:
                pass

        # Forensic finding: turn 7 was a CLEAN, on-time <ref> injection (no
        # timeout, no fallback) that was still followed by 50+ seconds of
        # pure silence -- proving the stuck-silence failure is not only the
        # timeout race. Rather than wait for the next question's VAD to
        # notice (10-50s later, per logs_1), watch for real speech resuming
        # within this many seconds of injection and log a loud, immediate
        # warning (and close the turn out) if it does not. This does not by
        # itself fix the underlying model behavior; it makes it fast to
        # detect and measure so the fix can be verified on real hardware.
        self.post_inject_watchdog_sec = float(post_inject_watchdog_sec)
        # How long outgoing audio stays masked after a <ref> injection, to
        # cover the codec's trailing rendition of the injected reference text
        # (see _finish_ref_injection). 0 disables the mask.
        self.ref_audio_drain_sec = max(0.0, float(ref_audio_drain_sec))
        # False by default on RunPod evidence: spelling numerals out inside the
        # <ref> is what turned $309.32 into a spoken "$39.32" in logs_4. See
        # search_helpers.normalize_for_speech's docstring for the A/B.
        self.spell_ref_numbers = bool(spell_ref_numbers)

        # "Thinking sound": played in place of the model's own audio ONLY while
        # an online search is actually running (see _start_thinking_sound and
        # its call sites). Never played on turns the model answers from its
        # own knowledge, nor while the router is still deciding.
        self.thinking_sound_pcm: np.ndarray | None = None
        self._thinking_sound_cursor = 0
        if thinking_sound_path:
            if Path(thinking_sound_path).is_file():
                try:
                    self.thinking_sound_pcm = load_audio_24k(thinking_sound_path)
                    print(
                        f"[liveTryPlasticity][search] thinking sound loaded: {thinking_sound_path} "
                        f"({self.thinking_sound_pcm.shape[0] / TARGET_SR:.2f}s)",
                        flush=True,
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    print(
                        f"[liveTryPlasticity][search] failed to load thinking sound "
                        f"{thinking_sound_path!r}: {e!r}\n{tb}",
                        flush=True,
                    )
                    self.conv_logger.error("thinking_sound_load", e, tb)
                    self.thinking_sound_pcm = None
            else:
                print(
                    f"[liveTryPlasticity][search] thinking sound path not found: {thinking_sound_path} "
                    f"-- will stay silent during the search instead",
                    flush=True,
                )

        self._install_graph_hidden_capture()

    def _install_graph_hidden_capture(self) -> None:
        lm_model = self.lm
        lm_gen = self.lm_gen
        if hasattr(lm_gen, "prepare_step_input") and hasattr(lm_gen, "process_transformer_output"):
            @torch.no_grad()
            def personaplex_step_with_hidden(
                self_gen,
                input_tokens: torch.Tensor = None,
                moshi_tokens: torch.Tensor = None,
                text_token: torch.Tensor = None,
                depformer_replace_tokens: torch.Tensor | None = None,
            ):
                prepared = self_gen.prepare_step_input(input_tokens, moshi_tokens, text_token)
                if prepared is None:
                    return None
                input_, provided_, target_, model_input_position, target_position = prepared
                state = self_gen._streaming_state
                transformer_out, text_logits = state.graphed_main(input_)
                output = self_gen.process_transformer_output(
                    transformer_out,
                    text_logits,
                    provided_,
                    target_,
                    model_input_position,
                    target_position,
                )
                return output, transformer_out, transformer_out

            lm_gen._step = types.MethodType(personaplex_step_with_hidden, lm_gen)
            # This override forwards text_token to prepare_step_input, and
            # process_transformer_output honors a provided token instead of
            # sampling one. That is what lets _step() force the model silent
            # while a search is in flight (see _step / suppress_text_until_ref).
            self._step_supports_text_token = True
            lm_gen.streaming_forever(1)
            self._warmup_runtime()
            print("[liveTryPlasticity] installed PersonaPlex graphed hidden capture", flush=True)
            return

        from moshi.models.lm import scatter_with_mask_
        from moshi.modules.transformer import create_sin_embedding
        from moshi.utils.sampling import sample_token

        capture_layer = int(self.tf_capture_layer) % len(lm_model.transformer.layers)

        old_state = getattr(lm_gen, "_streaming_state", None)
        if old_state is not None:
            with contextlib.suppress(Exception):
                old_state.__exit__(None, None, None)
            with contextlib.suppress(Exception):
                lm_gen._stop_streaming()

        def forward_text_with_layer(self_lm, sequence, sum_condition=None, cross_attention_src=None):
            B, K, S = sequence.shape
            assert K == self_lm.num_codebooks, (K, self_lm.num_codebooks)
            input_sequence = sequence
            input_ = None
            for cb_index in range(self_lm.num_audio_codebooks):
                audio_emb = self_lm.emb[cb_index](input_sequence[:, cb_index + self_lm.audio_offset])
                input_ = audio_emb if input_ is None else input_ + audio_emb
            text_emb = self_lm.text_emb(input_sequence[:, 0])
            input_ = text_emb if input_ is None else input_ + text_emb
            if sum_condition is not None:
                input_ = input_ + sum_condition.to(input_)
            if cross_attention_src is not None:
                cross_attention_src = cross_attention_src.to(input_)

            transformer = self_lm.transformer
            _, T, C = input_.shape
            dtype_input = input_.dtype
            state = transformer._streaming_state
            if state is None:
                offsets = torch.zeros(1, dtype=torch.long, device=input_.device)
            else:
                offsets = state.offsets

            x = input_
            if transformer.positional_embedding in {"sin", "sin_rope"}:
                positions = torch.arange(T, device=x.device).view(1, -1, 1)
                positions = positions + offsets.view(-1, 1, 1)
                pos_emb = create_sin_embedding(positions, C, max_period=transformer.max_period, dtype=x.dtype)
                x = x + transformer.positional_scale * pos_emb

            captured = x
            for idx, layer in enumerate(transformer.layers):
                x = layer(x, cross_attention_src=cross_attention_src)
                if idx == capture_layer:
                    captured = x

            if state is not None:
                state.offsets[:] = torch.where(state.exec_mask, state.offsets + T, state.offsets)

            transformer_out = x.to(dtype_input)
            layer_hidden = captured.to(dtype_input)
            if self_lm.out_norm:
                transformer_out = self_lm.out_norm(transformer_out)
            text_logits = self_lm.text_linear(transformer_out)
            text_logits = text_logits[:, None]
            return transformer_out, text_logits, layer_hidden

        @torch.no_grad()
        def step_with_layer(self_gen, input_tokens: torch.Tensor, depformer_replace_tokens: torch.Tensor | None = None):
            state = self_gen._streaming_state
            if state is None:
                raise RuntimeError("You should wrap those calls with a `with lm_gen.streaming(): ...`.")
            lm_model_local = self_gen.lm_model

            assert input_tokens.dim() == 3, "Shape should be [B, K, T]."
            B, Ki, S = input_tokens.shape
            assert B == state.batch_size, f"Got a batch size {B}, expected {state.batch_size}"
            assert S == 1, "Only support being given steps one by one."
            needed_tokens = lm_model_local.num_codebooks - lm_model_local.dep_q - 1
            assert Ki >= needed_tokens, f"We expect {needed_tokens} tokens from the user stream, got {Ki}."
            if Ki > needed_tokens:
                input_tokens = input_tokens[:, :needed_tokens, :]

            CT = state.cache.shape[2]
            delays = self_gen.delays_cuda[lm_model_local.dep_q + 1:]
            write_positions = (state.offsets[:, None, None] + delays[:, None]) % CT
            scatter_with_mask_(state.cache[:, lm_model_local.dep_q + 1:], -1, write_positions, input_tokens, state.exec_mask[:, None, None])

            is_init = state.offsets[:, None, None] <= self_gen.delays_cuda[:, None]
            is_init |= ~state.exec_mask[:, None, None]
            positions = (state.offsets % CT)[:, None, None].expand_as(is_init)
            input_ = state.cache.gather(dim=2, index=positions)
            input_ = torch.where(is_init, state.initial, input_)

            if self_gen.check:
                assert not (input_ == lm_model_local.ungenerated_token_id).any(), (state.offsets, input_)
                assert (input_[:, lm_model_local.audio_offset:] <= lm_model_local.card).all(), input_
                assert (input_[:, :1] <= lm_model_local.text_card).all()

            zero = torch.full((1,), lm_model_local.zero_token_id, dtype=torch.long, device=input_.device)
            if self_gen.cfg_coef != 1.:
                if state.cfg_is_masked_until is not None:
                    limit = self_gen.delays_cuda[:, None] + state.cfg_is_masked_until.view(-1, 1, 1)
                    is_zeroed = state.offsets[:, None, None] <= limit
                    masked = torch.where(is_zeroed & ~is_init, zero, input_)
                    input_ = torch.cat([input_, masked], dim=0)
                else:
                    input_ = input_.repeat(2, 1, 1)
                if self_gen.cfg_is_no_text:
                    input_[B:, :1] = torch.where(~is_init[:, :1], zero, input_[B:, :1])

            transformer_out, text_logits, layer_hidden = state.graphed_main(input_, state.condition_sum, state.condition_cross)
            if self_gen.cfg_coef != 1.:
                logits, logits_null = text_logits.chunk(2)
                if self_gen.cfg_is_no_text:
                    text_logits = logits
                    layer_hidden = layer_hidden[:B]
                else:
                    text_logits = logits_null + (logits - logits_null) * self_gen.cfg_coef
                    layer_hidden = layer_hidden[:B]

            if self_gen.on_text_logits_hook:
                self_gen.on_text_logits_hook(text_logits)
            text_token = sample_token(text_logits.float(), self_gen.use_sampling, self_gen.temp_text, self_gen.top_k_text)
            assert text_token.dim() == 3, text_token.shape
            assert text_token.shape[2] == 1
            assert text_token.shape[1] == 1, "Only one text stream supported."
            text_token = text_token[:, 0, 0]
            if self_gen.on_text_hook is not None:
                self_gen.on_text_hook(text_token)

            if state.graphed_depth is None:
                audio_tokens = None
            else:
                if depformer_replace_tokens is None:
                    audio_tokens = state.graphed_depth(text_token, transformer_out)
                else:
                    assert depformer_replace_tokens.dim() == 3
                    audio_tokens = depformer_replace_tokens.squeeze(-1)
                if self_gen.on_audio_hook is not None:
                    self_gen.on_audio_hook(audio_tokens)

            state.offsets = torch.where(state.exec_mask, state.offsets + 1, state.offsets)
            state.offset_cpu += 1
            positions = (state.offsets % CT)[:, None, None]
            scatter_with_mask_(state.cache[:, :1], -1, positions, text_token[:, None, None], state.exec_mask[:, None, None])
            if audio_tokens is not None:
                audio_tokens = audio_tokens[:, :, None]
                scatter_with_mask_(state.cache[:, 1: lm_model_local.dep_q + 1, :], -1, positions.expand_as(audio_tokens), audio_tokens, state.exec_mask[:, None, None])

            if not self_gen.support_out_of_sync and state.offset_cpu <= self_gen.max_delay:
                return None
            gen_delays_cuda = self_gen.delays_cuda[: lm_model_local.dep_q + 1]
            index = (state.offsets[:, None, None] - self_gen.max_delay + gen_delays_cuda[:, None]) % CT
            out = state.cache.gather(dim=2, index=index)
            mask = (state.offsets <= self_gen.max_delay) | ~state.exec_mask
            out[mask, :, :] = lm_model_local.ungenerated_token_id
            return out, transformer_out, layer_hidden

        lm_model.forward_text = types.MethodType(forward_text_with_layer, lm_model)
        lm_gen._step = types.MethodType(step_with_layer, lm_gen)
        # step_with_layer takes only input_tokens, so text cannot be forced on
        # this fallback path; search-window text suppression degrades to a no-op.
        self._step_supports_text_token = False
        lm_gen.streaming_forever(1)
        self._warmup_runtime()
        print(f"[liveTryPlasticity] installed graphed layer capture layer={self.tf_capture_layer}", flush=True)

    # -- Turn-detection constants (frame = one MIMI_FRAME_SIZE / 80ms chunk,
    # same granularity as _step() itself) --
    _VAD_SILENCE_FRAMES_REQUIRED = 12   # ~960ms of silence ends an utterance
    # Class-level fallback only -- __init__ always overrides this with an
    # instance attribute derived from the --search_max_filler_sec CLI flag.
    _SEARCH_MAX_FILLER_FRAMES = 25          # ~2s filler cap before a fallback <ref>
    # RMS level at which the model's own output counts as "it has started
    # speaking". Used for the time-to-first-word latency metric.
    _SPEECH_RMS_THRESHOLD = 0.006
    # Hard ceiling on buffered STT frames for ONE utterance (~60s at 80ms per
    # frame). Nobody speaks a single question for a minute; anything longer
    # means the VAD is stuck, and concatenating it would rebuild exactly the
    # runaway-transcript bug seen in logs_2 (see _stt_step).
    _STT_MAX_BUFFER_FRAMES = 750
    # Highest --inject_tokens_per_tick that has ever produced audible search
    # answers on real hardware. 4 delivered 79/72/97 packets (logs_3, logs_4);
    # 14 delivered 8/8/5 and was heard as total silence (logs_5). See the
    # warning in __init__.
    _INJECT_TOKENS_PER_TICK_SAFE_MAX = 6

    # Conservative default: _install_graph_hidden_capture() sets the real value
    # for whichever step override it installs. Only the PersonaPlex graphed
    # path forwards a text_token, so search-window text suppression is a no-op
    # on the fallback path rather than a TypeError.
    _step_supports_text_token = False

    def reset_session(self) -> None:
        # A session that ends (or is restarted by a reconnect) with a turn still
        # in flight would otherwise never have its latency block written -- the
        # block is normally emitted when the NEXT question arrives, and for the
        # last turn of a conversation there is no next question. Flush it here,
        # while the record and the response bounds are still intact. Guarded
        # with getattr: reset_session also runs during __init__ (via
        # _warmup_runtime), before any of this state exists.
        if getattr(self, "_latency_turn_id", None) is not None:
            try:
                import search_helpers

                tail = search_helpers.strip_injected_tags(
                    self.audio_text[self._turn_start_audio_text_len:]
                )
            except Exception:
                tail = ""
            self._finish_turn_latency(
                tail, outcome="session ended before the user asked anything else"
            )

        super().reset_session()
        # Per-conversation STT/turn-detection state. No-ops harmlessly when
        # STT isn't configured (self.stt_lm_gen stays None).
        self.stt_token_buffer: list = []
        self.stt_in_utterance = False
        self.stt_silence_frame_count = 0
        self.stt_last_vad_end = False
        # perf_counter reading of the moment the current utterance's first
        # real (non-padding) STT token arrived -- see _stt_step's buffer reset.
        self._utterance_started_perf: float | None = None
        self.search_turn_epoch = 0
        self.search_ref_committed_this_turn = False
        self.search_awaiting_ref = False
        # Set True if an uncaught exception ever escapes the per-chunk STT/
        # search hook in _step() -- once set, search is skipped for the rest
        # of THIS session but the underlying avatar conversation keeps working.
        self.search_hard_disabled = False
        # Per-turn stage-timing accumulator, see _start_turn / _log_timing_summary.
        self._turn_timing_start: float | None = None
        self._turn_timing_stages: dict[str, float] = {}
        # -- Dedicated end-to-end latency log (latency_logger.py) ------------
        self._turn_speech_end_perf: float | None = None
        # Stage timings/counts measured in _stt_step, before the turn record exists.
        self._pending_stt_stages: dict[str, float] = {}
        self._pending_stt_counts: dict[str, int] = {}
        # Which turn the per-chunk generation costs below are being charged to,
        # and when the model last emitted a text token.
        self._latency_turn_id: int | None = None
        self._last_text_emit_perf: float | None = None
        self._answer_end_perf: float | None = None
        self._turn_text_token_count = 0
        # Cross-thread handoff slots. The background thread WRITES these; the
        # GPU thread POLLS them once per 80ms chunk in _consume_pending() and
        # is the only thread allowed to touch self.lm_gen. `pending_lookup` is
        # separate from `pending_ref` because the <lookup> filler is only
        # injected AFTER the router has decided a search is happening.
        self.pending_lookup_tokens: list | None = None
        self.pending_ref_tokens: list | None = None
        self.pending_search_cancelled = False
        # Set by the background thread the moment it is about to hit the search
        # API; the GPU thread turns it into an actual _start_thinking_sound()
        # call on its next chunk.
        self.pending_start_thinking = False
        # While True, _step() forces the model's text stream to its own
        # zero_text_code so it composes NOTHING during a search. Muting the
        # outgoing audio alone is not enough: the model keeps generating text
        # behind the filler, so by the time the <ref> arrives it is already
        # mid-sentence with an invented figure and simply finishes it.
        self.suppress_text_until_ref = False
        self._pending_ref_token_counts = (0, 0)
        self.search_filler_frame_count = 0
        # Whether the thinking-sound suppression path was engaged for the turn
        # currently in flight -- read by turn_flags() when that turn's final
        # response is logged. Reset per turn in _start_turn / _begin_casual_turn.
        self._turn_used_thinking_sound = False
        self.search_session_history: list[tuple[str, str]] = []
        self.search_current_transcript = ""
        # Snapshot of len(self.audio_text) at turn start (when the user STOPPED
        # speaking), marking where this turn's assistant response begins.
        self._turn_start_audio_text_len = 0
        # Snapshot of len(self.audio_text) at the moment the user STARTS
        # speaking the next utterance, marking where this turn's response ends.
        self._utterance_start_audio_text_len = 0
        # True time-to-first-spoken-word tracking -- the only honest latency
        # metric for a full-duplex model.
        self._turn_awaiting_first_speech = False
        self._turn_first_speech_epoch = 0
        # Audio DELIVERY state, deliberately separate from the text/turn state
        # above. "The answer finished" and "the user heard the answer" are
        # different claims and logs_2 could not tell them apart.
        self._turn_first_audio_generated_perf: float | None = None
        self._turn_audio_stream_started = False
        self._turn_last_audio_out_perf: float | None = None
        self._turn_audio_packets_out = 0
        self.search_thinking_active = False
        self._thinking_sound_cursor = 0
        self._thinking_sound_started_at = 0.0
        self._thinking_sound_play_count = 0
        # Stop armed by _defer_thinking_sound_stop but not yet taken, as
        # (reason, reason_text, turn_id); None when nothing is armed.
        self._thinking_stop_pending: tuple[str, str, object] | None = None
        # Remaining 80ms steps for which outgoing audio stays masked after a
        # <ref> injection, so the codec's trailing rendition of the injected
        # reference text is never heard. See _finish_ref_injection.
        self._ref_audio_drain_remaining: int = 0
        # When suppression (forced silence) started, for the max_suppress_sec
        # cap in _consume_pending. None means "not currently suppressing".
        self._suppress_started_perf: float | None = None
        # Tokens still waiting to be fed in _inject_tokens, drained a few per
        # tick (inject_tokens_per_tick) instead of all at once, so injection
        # never blocks the real-time GPU thread for more than a fraction of a
        # tick's budget. (kind, remaining_tokens) or None when idle.
        self._injection_in_progress: tuple[str, list[int]] | None = None
        self._injection_started_perf: float = 0.0
        self._injection_total_tokens: int = 0
        self._injection_token_text: str = ""
        # Post-injection stuck-silence watchdog: set to the perf_counter
        # reading when a <ref>/fallback injection completes, cleared the
        # moment real speech (non-padding token) is observed again. If it
        # stays set past post_inject_watchdog_sec, _consume_pending logs a
        # loud warning and closes the turn out immediately instead of
        # waiting for the next question's VAD to notice.
        self._post_inject_watch_started: float | None = None
        self._post_inject_watch_turn: int | None = None
        self._post_inject_watch_fired = False

    def _next_thinking_sound_chunk(self) -> np.ndarray:
        """Next MIMI_FRAME_SIZE samples of the thinking sound, looping
        seamlessly. Advances self._thinking_sound_cursor and
        self._thinking_sound_play_count (incremented every time the clip
        wraps back to its start, i.e. every completed extra play-through)."""
        pcm = self.thinking_sound_pcm
        n = pcm.shape[0]
        out = np.empty(MIMI_FRAME_SIZE, dtype=np.float32)
        pos = 0
        cursor = self._thinking_sound_cursor % n
        while pos < MIMI_FRAME_SIZE:
            take = min(MIMI_FRAME_SIZE - pos, n - cursor)
            out[pos:pos + take] = pcm[cursor:cursor + take]
            pos += take
            new_cursor = (cursor + take) % n
            if new_cursor < cursor or (new_cursor == 0 and take > 0):
                self._thinking_sound_play_count += 1
            cursor = new_cursor
        self._thinking_sound_cursor = cursor
        return out

    def _will_actually_search(self) -> bool:
        """Whether a 'search' decision would really reach the network.

        The thinking sound is gated on this, not merely on the router saying
        'search': with web search disabled or no API key configured, the
        pipeline decides to search, discovers it cannot, and falls straight
        back to the model's own knowledge. Playing a 'searching...' cue for a
        search that never happens would tell the user something untrue."""
        return bool(self.web_search_enabled and self.web_search_api_key)

    def _start_thinking_sound(self, turn_id, transcript: str = "") -> None:
        """Begin the thinking sound. MUST run on the GPU thread -- it mutates
        the playback state that _step() reads every 80ms.

        Idempotent: the rules path starts the sound synchronously the instant
        it commits to searching, while the model-router path signals from the
        background thread, and both can land for the same turn."""
        if self.suppress_text_during_search:
            if not self.suppress_text_until_ref:
                self._suppress_started_perf = time.perf_counter()
            self.suppress_text_until_ref = True
        # Recorded regardless of thinking_sound_pcm below: this flag answers
        # "did this turn wait on a search" for the conversation log, which is
        # true even when there is no audio clip configured to cover the wait.
        self._turn_used_thinking_sound = True
        if self.thinking_sound_pcm is None or self.search_thinking_active:
            return
        self.search_thinking_active = True
        self._thinking_stop_pending = None
        self._ref_audio_drain_remaining = 0
        self._thinking_sound_started_at = time.perf_counter()
        self._thinking_sound_play_count = 1
        self.conv_logger.event(
            "thinking_sound_start",
            f"THINKING_SOUND START turn_id={turn_id} reason=web_search_required",
            transcript=transcript,
        )
        self.conv_logger.narrate_thinking_start(turn_id)

    def _defer_thinking_sound_stop(self, turn_id, reason: str, reason_text: str) -> None:
        """Arm the stop instead of stopping now.

        The ref tokens are about to be force-fed; PersonaPlex's audio codebooks
        trail its text stream, so for a short window after the injection the
        decoder is still emitting the acoustic realisation of the injected
        reference (logs_2 turn 6: the assistant audibly said ". Class ( stock
        $. reflecting. move opened>" before its real answer). The assistant is
        therefore NOT yet "ready to produce the real answer" at this point --
        the thinking sound keeps covering the wait, and the real stop fires
        from _step() once _ref_audio_drain_remaining reaches zero."""
        if not self.search_thinking_active:
            return
        self._thinking_stop_pending = (str(reason), str(reason_text), turn_id)

    def _stop_thinking_sound(self, turn_id, reason: str, reason_text: str) -> None:
        """Shared stop logic for both the real-ref-ready and filler-timeout
        paths in _consume_pending -- keeps duration/loop-count bookkeeping in
        one place."""
        if not self.search_thinking_active:
            return
        self.search_thinking_active = False
        self._thinking_stop_pending = None
        self._ref_audio_drain_remaining = 0
        duration_s = max(0.0, time.perf_counter() - self._thinking_sound_started_at)
        clip_duration_s = (
            self.thinking_sound_pcm.shape[0] / TARGET_SR if self.thinking_sound_pcm is not None else 0.0
        )
        self.conv_logger.event(
            "thinking_sound_stop",
            f"THINKING_SOUND STOP duration={duration_s:.3f}s turn_id={turn_id} reason={reason}",
            reason=reason, duration_s=round(duration_s, 3),
            play_count=self._thinking_sound_play_count,
        )
        self.conv_logger.narrate_thinking_stop(
            turn_id, reason_text, duration_s, self._thinking_sound_play_count, clip_duration_s
        )
        self.conv_logger.latency.stage(
            getattr(self, "_latency_turn_id", None), "thinking_sound", duration_s,
            note=f"stopped because {reason}, played {self._thinking_sound_play_count}x",
        )

    def note_audio_streamed(self, packet_rms: float) -> None:
        """Called by the websocket audio sender the instant a packet is written
        to the socket -- the ONLY point in the pipeline that can honestly say
        the user was sent audio.

        Runs on the asyncio loop, not the GPU thread. It only touches plain
        attributes and LatencyLogger.mark/count, which take their own lock, so
        it needs no lock of its own and cannot block the sender. Silent packets
        (including the thinking sound's own frames, which are gated out
        upstream by force_idle) are ignored so the mark means 'the answer
        started reaching the user', not 'a packet existed'."""
        try:
            if packet_rms < self._SPEECH_RMS_THRESHOLD:
                return
            now = time.perf_counter()
            self._turn_last_audio_out_perf = now
            self._turn_audio_packets_out += 1
            if self._turn_audio_stream_started:
                return
            self._turn_audio_stream_started = True
            turn_id = self._latency_turn_id
            if turn_id is None:
                return
            self.conv_logger.latency.mark(
                turn_id, "audio_stream_started",
                "first audio packet written to the websocket", at=now,
            )
            generated = self._turn_first_audio_generated_perf
            if generated is not None:
                self.conv_logger.latency.count(
                    turn_id,
                    generated_to_sent_s=round(max(0.0, now - generated), 3),
                )
        except Exception:
            pass

    def _inject_tokens(self, tokens: list[int]) -> None:
        """Force-feed text tokens into the live stream. reset_streaming() is
        never called: the shared KV-cache (and the conversation heard so far)
        stays intact across the injection.

        Calls `self.lm_gen._step(...)` -- the SAME patched, hidden-capturing
        method every other 80ms tick in _step() uses -- rather than the public
        `self.lm_gen.step(...)` wrapper. Both used to resolve to the same
        underlying call (Python attribute lookup finds the instance-level
        monkey-patch installed by _install_graph_hidden_capture() either way),
        but `.step()`'s own bookkeeping was written for a `_step()` that
        returns the library's stock shape, not our 3-tuple
        (output, transformer_out, transformer_out). Calling `._step()`
        directly removes that mismatch and keeps injection on the exact same
        code path as normal generation, which is the most likely source of
        the intermittent "model goes silent after an injection and never
        recovers" failures seen in logs_1/conversation.log (turn 7: a clean,
        on-time <ref> injection followed by 50+ seconds of pure silence).

        Only safe when the installed `_step` actually accepts these kwargs
        (the PersonaPlex graphed-hidden path -- see
        _install_graph_hidden_capture / _step_supports_text_token). The
        fallback (non-PersonaPlex) graphed-layer path's `_step` takes a
        positional `input_tokens` tensor only, so injection there still goes
        through the public `.step()` wrapper, exactly as before."""
        for tok in tokens:
            if self._step_supports_text_token:
                self.lm_gen._step(
                    moshi_tokens=self.lm_gen._encode_zero_frame(),
                    text_token=tok,
                    input_tokens=self.lm_gen._encode_sine_frame(),
                )
            else:
                self.lm_gen.step(
                    moshi_tokens=self.lm_gen._encode_zero_frame(),
                    text_token=tok,
                    input_tokens=self.lm_gen._encode_sine_frame(),
                )

    def _stt_step(self, chunk: torch.Tensor) -> None:
        """Run the separate STT/VAD submodel one 80ms frame forward (same GPU
        thread as everything else in _step()); on a detected end-of-utterance,
        kick off a turn. Never touches self.lm_gen/self.mimi."""
        stt_codes = self.stt_mimi.encode(chunk)
        stt_result = self.stt_lm_gen.step_with_extra_heads(stt_codes)
        if stt_result is None:
            return
        stt_tokens, vad_heads = stt_result
        vad_score = 0.0
        if vad_heads and len(vad_heads) > 2:
            vad_score = float(vad_heads[2][0, 0, 0].cpu().item())
        if stt_tokens is not None:
            frame_tokens = stt_tokens[:, :1, :].cpu()
            text_token = stt_tokens[0, 0, 0].item()
            if text_token not in (0, self.stt_padding_token_id):
                if not self.stt_in_utterance:
                    # First real word of a new user utterance: everything the
                    # model emitted before this point belongs to the PREVIOUS
                    # turn's response, everything after is it reacting to what
                    # it is hearing now.
                    self._utterance_start_audio_text_len = len(self.audio_text)
                    self._answer_end_perf = self._last_text_emit_perf
                    # CRITICAL (forensic fix, logs_2/conversation.log): drop
                    # everything buffered BEFORE this utterance began. The
                    # buffer used to be cleared only after a turn completed,
                    # so every frame emitted while the assistant was speaking
                    # -- 50+ seconds of it in that log (stt_frames_decoded
                    # grew 226 -> 716) -- was concatenated into the next
                    # transcript. That is what turned "What is today's Tesla
                    # stock market price?" into "Life is, today is, test
                    # life, stock market price." and then sent that garbage
                    # to the search API as the query. Only the current
                    # utterance's frames may reach decode_stt_tokens().
                    dropped = len(self.stt_token_buffer)
                    self.stt_token_buffer = []
                    self._utterance_started_perf = time.perf_counter()
                    if dropped:
                        self.conv_logger.latency.count(
                            self._latency_turn_id, stt_pre_utterance_frames_dropped=dropped,
                        )
                self.stt_in_utterance = True
                self.stt_last_vad_end = False
            self.stt_token_buffer.append(frame_tokens)
            # Safety net: even inside one utterance the buffer must not grow
            # without bound (a stuck VAD would otherwise rebuild the same
            # runaway-concatenation bug). 60s of frames is far longer than
            # any real spoken question.
            if len(self.stt_token_buffer) > self._STT_MAX_BUFFER_FRAMES:
                self.stt_token_buffer = self.stt_token_buffer[-self._STT_MAX_BUFFER_FRAMES:]
        if vad_score > self.vad_threshold:
            self.stt_silence_frame_count += 1
        else:
            self.stt_silence_frame_count = 0
        vad_fired = (
            self.stt_silence_frame_count >= self._VAD_SILENCE_FRAMES_REQUIRED
            and not self.stt_last_vad_end
        )
        self.stt_last_vad_end = self.stt_silence_frame_count >= self._VAD_SILENCE_FRAMES_REQUIRED
        # Second layer of the same forensic fix: while we are definitively
        # NOT inside a user utterance and the VAD says the floor is silent,
        # nothing buffered can belong to a future question. Anything the STT
        # emits here is echo of the assistant's own voice through the user's
        # speakers, or a hallucination on room noise -- both of which are
        # non-padding tokens that survive decode_stt_tokens()'s padding
        # filter, which is exactly how 50s of them ended up prepended to a
        # real question in logs_2.
        if (
            not self.stt_in_utterance
            and self.stt_token_buffer
            and self.stt_silence_frame_count >= self._VAD_SILENCE_FRAMES_REQUIRED
        ):
            self.stt_token_buffer = []
        if vad_fired and self.stt_in_utterance and self.stt_token_buffer and not self.search_awaiting_ref:
            # Since injection is now spread across several ticks
            # (search_awaiting_ref goes False the instant it STARTS, not when
            # it finishes -- see _consume_pending), a new utterance's VAD can
            # fire while a few <ref>/fallback tokens from the PREVIOUS turn
            # are still queued. Flush them synchronously right here rather
            # than either (a) blocking vad_fired -- which would leave that
            # utterance's tokens sitting unflushed in stt_token_buffer,
            # merging it with whatever the user says next once the block
            # lifts -- or (b) letting _start_turn() proceed while the old
            # injection is still trickling in, interleaving the previous
            # turn's forced tokens with the new turn's own generation. What
            # is left at this point is only the last few tokens of that
            # block (most of it already fed on prior ticks), so this is a
            # short, bounded flush, not a return of the original ~1.3s stall.
            if self._injection_in_progress is not None:
                kind, remaining = self._injection_in_progress
                self._injection_in_progress = None
                self._inject_tokens(remaining)
                self._finish_ref_injection(kind, time.perf_counter() - self._injection_started_perf)
            import search_helpers

            t_speech_end = time.perf_counter()
            n_stt_frames = len(self.stt_token_buffer)
            t_stt0 = time.perf_counter()
            transcript, transcript_token_ids = search_helpers.decode_stt_tokens_with_ids(
                self.stt_token_buffer, self.stt_tokenizer, self.stt_padding_token_id
            )
            stt_decode_elapsed = time.perf_counter() - t_stt0
            self.stt_token_buffer = []
            self.stt_in_utterance = False
            self.stt_silence_frame_count = 0
            if transcript.strip():
                # Log the PREVIOUS turn's assistant response now that we know
                # it's finished (the user has started speaking again).
                resp_end = self._utterance_start_audio_text_len
                if resp_end <= self._turn_start_audio_text_len:
                    resp_end = len(self.audio_text)
                prev_response = search_helpers.strip_injected_tags(
                    self.audio_text[self._turn_start_audio_text_len:resp_end]
                )
                if prev_response:
                    self.conv_logger.turn_replied(self.search_turn_epoch, prev_response)
                    self.conv_logger.assistant_response(self.search_current_transcript, prev_response)
                    self.conv_logger.narrate_response(
                        self.search_turn_epoch, self.search_current_transcript, prev_response
                    )
                    # avatar_streaming_active is True here by construction: this
                    # hook only runs inside the reply-generation engine that
                    # feeds the avatar/audio pipeline, never in a code path
                    # without it. thinking_sound_played reflects whether this
                    # turn actually covered a wait with the filler clip.
                    self.conv_logger.turn_flags(
                        self.search_turn_epoch,
                        ref_lora_active=bool(getattr(self, "ref_lora_dir", "")),
                        avatar_streaming_active=True,
                        thinking_sound_played=bool(getattr(self, "_turn_used_thinking_sound", False)),
                    )
                    self._finish_turn_latency(prev_response)
                    if self.search_current_transcript:
                        self.search_session_history.append((self.search_current_transcript, prev_response))
                        self.search_session_history = self.search_session_history[-6:]
                elif self.search_current_transcript:
                    turn_start_perf = self._turn_timing_start
                    gap_s = max(0.0, time.perf_counter() - turn_start_perf) if turn_start_perf else 0.0
                    self.conv_logger.no_response_warning(
                        self.search_turn_epoch, self.search_current_transcript, gap_s
                    )
                    self.conv_logger.narrate_no_response_warning(
                        self.search_turn_epoch, self.search_current_transcript, gap_s
                    )
                    self._finish_turn_latency("", outcome="no spoken response was produced")
                # -- Transcript sanity gate --------------------------------
                t_check0 = time.perf_counter()
                usable, script_stats = search_helpers.check_transcript_usable(
                    transcript,
                    self.stt_max_non_latin_ratio,
                    require_english=self.stt_require_english,
                )
                transcript_check_elapsed = time.perf_counter() - t_check0
                if not usable and self.stt_reject_foreign_script:
                    self.conv_logger.turn_transcript_rejected(
                        self.search_turn_epoch + 1, transcript, script_stats, transcript_token_ids
                    )
                    hint = ""
                    if script_stats.get("kind") == "script":
                        hint = (
                            f"\n                        first ids={transcript_token_ids[:24]} -- if "
                            f"these look like ordinary ids, the STT tokenizer is likely mismatched; "
                            f"compare the 'stt tokenizer=' line printed at startup."
                        )
                    print(
                        f"[liveTryPlasticity][STT] rejected transcript "
                        f"({script_stats.get('reason', 'unusable')}, "
                        f"{len(transcript_token_ids)} tok): {transcript[:120]!r}{hint}",
                        flush=True,
                    )
                    with contextlib.suppress(Exception):
                        self.stt_lm_gen.reset_streaming()
                        self.stt_mimi.reset_streaming()
                    return

                self.conv_logger.turn_heard(
                    self.search_turn_epoch + 1, transcript, transcript_token_ids, script_stats
                )
                self.conv_logger.narrate_user_message(self.search_turn_epoch + 1, transcript)
                self._turn_speech_end_perf = t_speech_end
                self._pending_stt_stages = {
                    "stt_decode": stt_decode_elapsed,
                    "transcript_check": transcript_check_elapsed,
                }
                # stt_utterance_sec is the headline STT health number: it is
                # how much audio actually went into this transcript. Before
                # the buffer fix it silently grew to ~57s (logs_2), which is
                # what corrupted the transcripts; it should now track the
                # length of the spoken question (a few seconds).
                utt_started = getattr(self, "_utterance_started_perf", None)
                self._pending_stt_counts = {
                    "stt_frames_decoded": n_stt_frames,
                    "stt_utterance_sec": round(n_stt_frames * MIMI_FRAME_SIZE / TARGET_SR, 2),
                    "stt_utterance_wall_sec": (
                        round(t_speech_end - utt_started, 2) if utt_started else None
                    ),
                    "transcript_tokens": len(transcript_token_ids),
                    "transcript_chars": len(transcript),
                }
                with contextlib.suppress(Exception):
                    self.stt_lm_gen.reset_streaming()
                    self.stt_mimi.reset_streaming()
                self._start_turn(transcript)

    def _begin_casual_turn(self, transcript: str, reason: str) -> None:
        """Mark a turn that will be answered from the model's own knowledge:
        no search, no injection, nothing added to the context."""
        self.search_turn_epoch += 1
        self.search_current_transcript = transcript
        self._turn_used_thinking_sound = False
        self._turn_start_audio_text_len = len(self.audio_text)
        self._turn_awaiting_first_speech = True
        self._turn_first_speech_epoch = self.search_turn_epoch
        print(
            f"[liveTryPlasticity][search] no search ({reason}) -- "
            f"answering from the model's own knowledge",
            flush=True,
        )

    def _start_turn(self, transcript: str) -> None:
        """Decide how to answer this utterance.

        Tier 0 (here, on the GPU thread): pure-regex routing, microseconds. A
        confident rule verdict either starts the search immediately or ends the
        turn as a casual one, with no model call at all.

        Tier 1 (background thread): anything the rules could not resolve is
        scored by the Qwen router in `_route_and_search`, which is tens of
        milliseconds -- far too slow for _step()'s 80ms budget."""
        import search_helpers

        self._turn_timing_start = time.perf_counter()
        self._turn_timing_stages = {}
        self._open_turn_latency(self.search_turn_epoch + 1, transcript)

        if not getattr(self, "search_enabled", False):
            self._begin_casual_turn(transcript, "search not configured")
            self.conv_logger.latency.mark(
                self._latency_turn_id, "decision_made", "search not configured"
            )
            return

        t_rules0 = time.perf_counter()
        ruled, rule_reason = search_helpers.rule_route_explain(transcript)
        rules_elapsed = time.perf_counter() - t_rules0
        self._turn_timing_stages["rule_route"] = rules_elapsed
        self.conv_logger.quick_gate_timing(rules_elapsed)
        self.conv_logger.latency.stage(
            self._latency_turn_id, "rule_route", rules_elapsed,
            note=("resolved by rules" if ruled is not None else "undecided, router will run"),
        )

        if ruled is False:
            self.conv_logger.turn_decision(
                self.search_turn_epoch + 1, transcript, needs_search=False, source="rules",
                score=0.0, elapsed_s=rules_elapsed,
                reason=rule_reason,
            )
            self.conv_logger.narrate_router_decision(
                self.search_turn_epoch + 1, transcript, False, "rules", 0.0, rule_reason,
            )
            self._begin_casual_turn(transcript, "rule: static phrase")
            self.conv_logger.turn_done(
                self.search_turn_epoch, "answered from own knowledge",
                time.perf_counter() - self._turn_timing_start,
            )
            self.conv_logger.latency.mark(
                self._latency_turn_id, "decision_made", "rules: no search needed"
            )
            self.conv_logger.latency.count(self._latency_turn_id, decided_by="rules", searched=False)
            return

        # Either the rules demanded a search (ruled is True) or they could not
        # decide (None) and the router will settle it on the background thread.
        self.search_turn_epoch += 1
        my_epoch = self.search_turn_epoch
        self.search_ref_committed_this_turn = False
        self.search_awaiting_ref = True
        self.search_filler_frame_count = 0
        self._turn_used_thinking_sound = False
        self.search_current_transcript = transcript
        self.pending_lookup_tokens = None
        self.pending_ref_tokens = None
        self.pending_search_cancelled = False
        self._turn_start_audio_text_len = len(self.audio_text)
        self._turn_awaiting_first_speech = True
        self._turn_first_speech_epoch = my_epoch

        if ruled is True:
            if self._will_actually_search():
                self._start_thinking_sound(my_epoch, transcript)
            self.conv_logger.turn_decision(
                my_epoch, transcript, needs_search=True, source="rules", score=1.0,
                elapsed_s=rules_elapsed,
                reason=rule_reason,
            )
            self.conv_logger.narrate_router_decision(
                my_epoch, transcript, True, "rules", 1.0, rule_reason,
            )
            self.conv_logger.latency.mark(
                self._latency_turn_id, "decision_made", "rules: search online"
            )
            self.conv_logger.latency.count(self._latency_turn_id, decided_by="rules", searched=True)

        threading.Thread(
            target=self._route_and_search,
            args=(transcript, my_epoch, ruled is True),
            daemon=True,
            name="query-search",
        ).start()

    def _route_and_search(self, transcript: str, my_epoch: int, rules_said_search: bool) -> None:
        """Background thread: (optionally) run the Qwen router, then -- only if
        a search is warranted -- web search, clean, and compress into a short
        grounding statement. Only touches the router/compressor objects and
        plain attributes; never self.lm_gen/self.mimi/self.stt_lm_gen. Results
        are handed to the GPU thread through self.pending_* slots, which
        _consume_pending() polls once per chunk."""
        import search_helpers

        try:
            if not rules_said_search:
                verdict = self.query_router.decide(transcript)
                self._turn_timing_stages["router"] = float(verdict.get("elapsed_s", 0.0))
                self.conv_logger.latency.stage(
                    my_epoch, "router_model", float(verdict.get("elapsed_s", 0.0)),
                    note=f"{verdict['source']}: score {float(verdict['score']):.3f}",
                )
                self.conv_logger.latency.mark(
                    my_epoch, "decision_made",
                    "search online" if verdict["needs_search"] else "answer directly",
                )
                self.conv_logger.latency.count(
                    my_epoch,
                    decided_by=str(verdict["source"]),
                    searched=bool(verdict["needs_search"]),
                    router_score=round(float(verdict["score"]), 4),
                    router_prompt_tokens=int(getattr(self.query_router, "last_prompt_tokens", 0)),
                )
                self.conv_logger.turn_decision(
                    my_epoch, transcript,
                    needs_search=bool(verdict["needs_search"]),
                    source=str(verdict["source"]),
                    score=float(verdict["score"]),
                    elapsed_s=float(verdict["elapsed_s"]),
                    reason=str(verdict["reason"]),
                )
                self.conv_logger.narrate_router_decision(
                    my_epoch, transcript, bool(verdict["needs_search"]),
                    str(verdict["source"]), float(verdict["score"]), str(verdict["reason"]),
                )
                print(
                    f"[liveTryPlasticity][router] needs_search={verdict['needs_search']} "
                    f"src={verdict['source']} score={verdict['score']:.3f} "
                    f"in {1000.0 * float(verdict['elapsed_s']):.0f}ms :: {transcript!r}",
                    flush=True,
                )
                if not verdict["needs_search"]:
                    self.conv_logger.turn_done(
                        my_epoch, "answered from own knowledge",
                        time.perf_counter() - (self._turn_timing_start or time.perf_counter()),
                    )
                    self._log_timing_summary()
                    if my_epoch == self.search_turn_epoch:
                        self.pending_search_cancelled = True
                    return

            if my_epoch != self.search_turn_epoch or self.search_ref_committed_this_turn:
                return

            # --- A search is happening: only now does the model get told to
            # wait. Injection itself must happen on the GPU thread. ---
            lookup_text = search_helpers.wrap_with_lookup_tags()
            self.pending_lookup_tokens = self.tokenizer.encode(lookup_text)

            hits: list[dict] = []
            if not self.web_search_enabled:
                print(
                    "[liveTryPlasticity][search] a search was needed but web search is "
                    "disabled -- falling back to the model's own knowledge",
                    flush=True,
                )
            elif not self.web_search_api_key:
                print(
                    "[liveTryPlasticity][search] a search was needed but no API key is "
                    "configured -- falling back to the model's own knowledge",
                    flush=True,
                )
            else:
                self.pending_start_thinking = True
                self.conv_logger.event(
                    "web_search_start", query=transcript, provider=self.web_search_provider,
                    triggered_reason="the router decided this needs current information",
                )
                self.conv_logger.narrate_web_search_start(
                    my_epoch, transcript, self.web_search_provider,
                    "the question needs current information",
                )
                t_web0 = time.perf_counter()
                web_hits = search_helpers.web_search_query_sync(
                    transcript, self.web_search_api_key, self.web_search_provider,
                    self.web_search_max_results, self.web_search_timeout,
                )
                web_elapsed = time.perf_counter() - t_web0
                # web_search_query/_sync always return [] on any failure (bad
                # response, exception, or timeout) rather than raising -- see
                # search_helpers.py -- so "why zero results" is inferred here
                # from elapsed time against the configured timeout, not a
                # distinct error channel.
                if web_hits:
                    search_status = "success"
                elif web_elapsed >= self.web_search_timeout * 0.95:
                    search_status = "timeout"
                else:
                    search_status = "empty_results"
                self._turn_timing_stages["web_search"] = web_elapsed
                self.conv_logger.latency.stage(
                    my_epoch, "web_search", web_elapsed,
                    note=f"{self.web_search_provider}: {len(web_hits)} result(s), status={search_status}",
                )
                self.conv_logger.latency.mark(my_epoch, "search_done")
                self.conv_logger.web_search(
                    transcript, self.web_search_provider, len(web_hits), web_elapsed,
                    triggered_reason="router decided live data was required",
                )
                runtime_logging.log_event(
                    runtime_logging.get_system_logger(), "WebSearch", "query_complete",
                    level=(logging.WARNING if search_status != "success" else logging.INFO),
                    provider=self.web_search_provider, status=search_status,
                    n_results=len(web_hits), elapsed_s=round(web_elapsed, 3),
                )
                t_filter0 = time.perf_counter()
                relevant = [h for h in web_hits if h.get("similarity_score", 0.0) >= self.web_search_min_score]
                sorted_hits = sorted(relevant, key=lambda c: c["similarity_score"], reverse=True)
                hits = sorted_hits[: self.web_search_max_results]
                filter_elapsed = time.perf_counter() - t_filter0
                self._turn_timing_stages["search_filter"] = filter_elapsed
                self.conv_logger.latency.stage(
                    my_epoch, "search_filter", filter_elapsed,
                    note=f"{len(web_hits)} found -> {len(hits)} kept",
                )
                self.conv_logger.latency.count(
                    my_epoch,
                    web_results_found=len(web_hits),
                    web_results_kept=len(hits),
                    web_result_chars=sum(len(str(h.get("text", ""))) for h in hits),
                )
                self.conv_logger.turn_search(
                    my_epoch, self.web_search_provider, len(web_hits), len(hits), web_elapsed,
                    status=search_status,
                )
                self.conv_logger.retrieval(transcript, "web", hits, web_elapsed)
                self.conv_logger.narrate_web_search_results(
                    my_epoch, sorted_hits, hits, self.web_search_max_results
                )
                if web_hits and not relevant:
                    print(
                        f"[liveTryPlasticity][search] all {len(web_hits)} web result(s) scored below "
                        f"web_search_min_score={self.web_search_min_score} -- discarding them and "
                        f"falling back to the model's own knowledge",
                        flush=True,
                    )

            if my_epoch != self.search_turn_epoch or self.search_ref_committed_this_turn:
                return

            # -- Compression: extractive-first, LLM only when extraction is
            # not confident enough. -----------------------------------------
            #
            # Forensic finding (logs_1/conversation.log, 2026-09-05 RunPod
            # run): the LLM compressor (Qwen2.5-1.5B, 4-bit) took 2.0-5.0s for
            # a 14-35 token reply on an RTX 5090 -- 10-30x slower than a small
            # model doing greedy decode should take in isolation. It shares
            # the GPU with the continuously-running PersonaPlex/IMTalker
            # avatar pipeline (25fps rendering + audio generation on the same
            # physical device from a different thread), which is the far more
            # likely explanation than the model itself being slow. That 2-5s
            # is by far the largest component of every search turn's latency,
            # AND it is what was racing (and losing to) the 6.0s filler
            # timeout in turn 4, discarding a correctly-computed answer.
            #
            # Most of these queries (price/score/rate lookups) are answered
            # by a single, mostly-clean sentence already present in the web
            # result -- exactly what extract_best_sentence() (pure Python,
            # ~0ms, no GPU) picks out. Try that FIRST; only pay for the LLM
            # forward pass when extraction is not confident (no digit in the
            # picked sentence, or a weak keyword-overlap score).
            grounding = ""
            used_fallback = False
            grounding_source = ""
            t_compress0 = time.perf_counter()

            extractive_text = ""
            extractive_score = 0.0
            if hits and self.compressor_mode in ("extractive_first", "extractive_only"):
                extractive_text, extractive_score = search_helpers.extract_best_sentence(transcript, hits)
                has_digit = bool(re.search(r"\d", extractive_text))
                # Almost every search-triggering question here asks for a
                # quantifiable value (price/rate/score/"how much") -- the
                # router's own live-topic rules are built around that. A
                # smoke test caught a real failure mode of an earlier, looser
                # version of this gate: a short, topically-related sentence
                # that did NOT contain the actual figure (e.g. "the gold
                # market can go through periods of quiet trading" for a gold
                # PRICE question) scored high enough on overlap alone to pass
                # as confident. Requiring a digit at the normal threshold, or
                # a much higher bar without one (name/event answers, e.g.
                # "who is the current president of France"), avoids that.
                confident = bool(
                    extractive_text
                    and (
                        (has_digit and extractive_score >= self.extractive_confidence_threshold)
                        or (not has_digit and extractive_score >= max(2 * self.extractive_confidence_threshold, 0.9))
                    )
                )
                if confident:
                    grounding = extractive_text
                    grounding_source = "extractive"
                    extractive_elapsed = time.perf_counter() - t_compress0
                    self._turn_timing_stages["compression"] = (
                        self._turn_timing_stages.get("compression", 0.0) + extractive_elapsed
                    )
                    self.conv_logger.compressor_call(
                        transcript, [h.get("text", "") for h in hits[:2]], grounding,
                        extractive_elapsed, used_fallback=False,
                    )
                    self.conv_logger.latency.stage(
                        my_epoch, "compression_extractive", extractive_elapsed,
                        note=f"score={extractive_score:.2f} has_digit={has_digit} (LLM call skipped)",
                    )
                    self.conv_logger.latency.count(
                        my_epoch, grounding_source="extractive",
                        extractive_score=round(extractive_score, 3), extractive_has_digit=has_digit,
                    )
                    self.conv_logger.narrate_summary(my_epoch, "a web search", len(hits), grounding, used_fallback=False)
                else:
                    runtime_logging.log_event(
                        runtime_logging.get_system_logger(), "Compressor", "extractive_not_confident",
                        score=round(extractive_score, 3), has_digit=has_digit,
                        text_preview=extractive_text[:80],
                    )

            if (
                not grounding and hits and self.context_compressor is not None
                and self.compressor_mode != "extractive_only"
            ):
                t_llm0 = time.perf_counter()
                grounding = self.context_compressor.compress(question=transcript, chunks=hits)
                grounding_source = "llm" if grounding else grounding_source
                primary_elapsed = time.perf_counter() - t_llm0
                self._turn_timing_stages["compression"] = (
                    self._turn_timing_stages.get("compression", 0.0) + primary_elapsed
                )
                self.conv_logger.compressor_call(
                    transcript, [h.get("text", "") for h in hits[:2]], grounding,
                    primary_elapsed, used_fallback=False,
                )
                comp_stats = dict(getattr(self.context_compressor, "last_stats", {}) or {})
                self.conv_logger.latency.stage(
                    my_epoch, "compression_llm", primary_elapsed,
                    note=(
                        f"{comp_stats.get('compressor_output_tokens', 0)} token(s) out"
                        + (" [REJECTED]" if comp_stats.get("compressor_rejected") else "")
                    ),
                )
                if comp_stats:
                    self.conv_logger.latency.count(my_epoch, **comp_stats)
                if grounding:
                    self.conv_logger.narrate_summary(my_epoch, "a web search", len(hits), grounding, used_fallback=False)

            if not grounding and hits:
                used_fallback = True
                grounding_source = grounding_source or "multi_sentence_fallback"
                t_fallback0 = time.perf_counter()
                grounding = search_helpers.summarize_web_fallback(
                    transcript, hits, max_sentences=2, max_chars=200
                )
                fallback_elapsed = time.perf_counter() - t_fallback0
                self._turn_timing_stages["compression"] = (
                    self._turn_timing_stages.get("compression", 0.0) + fallback_elapsed
                )
                self.conv_logger.compressor_call(
                    transcript, [h.get("text", "") for h in hits[:2]], grounding,
                    fallback_elapsed, used_fallback=True,
                )
                self.conv_logger.latency.stage(
                    my_epoch, "compression_fallback", fallback_elapsed,
                    note=f"{len(grounding)} char(s) out",
                )
                if grounding:
                    self.conv_logger.narrate_summary(my_epoch, "a web search", len(hits), grounding, used_fallback=True)
            self.conv_logger.latency.count(my_epoch, grounding_source=grounding_source or "none")
            if not grounding:
                self.conv_logger.narrate_no_information(my_epoch)
            self.conv_logger.turn_ground(my_epoch, grounding, used_fallback=used_fallback, source=grounding_source)
            self.conv_logger.latency.mark(
                my_epoch, "grounding_ready",
                {
                    "extractive": "extracted directly from search results (no LLM call)",
                    "llm": "compressed by the LLM",
                    "multi_sentence_fallback": "extractive fallback (LLM produced nothing usable)",
                }.get(grounding_source, "nothing usable was produced"),
            )

            if my_epoch != self.search_turn_epoch or self.search_ref_committed_this_turn:
                return

            # Rewrite the raw grounding fact into plain spoken English before
            # PersonaPlex ever sees it -- "$309.32" becomes "three hundred
            # nine dollars and thirty-two cents", "(TSLA)" is dropped, "+0.23%"
            # becomes "up zero point two three percent". Deterministic
            # (no LLM call, no added latency), and this is the ONE place all
            # three grounding sources (extractive/llm/fallback) funnel
            # through, so it applies uniformly. `grounding` above (already
            # logged via turn_ground/GROUND) stays the raw fact for
            # comparison; only the text actually injected is rewritten.
            raw_grounding = grounding
            spoken_grounding = (
                search_helpers.normalize_for_speech(grounding, spell_numbers=self.spell_ref_numbers)
                if grounding else grounding
            )
            if spoken_grounding != raw_grounding:
                self.conv_logger.event(
                    "spoken_summary",
                    f"SPOKEN_SUMMARY turn={my_epoch}",
                    raw_summary=raw_grounding, spoken_summary=spoken_grounding,
                )
            grounding = spoken_grounding

            ref_content = grounding.strip() if grounding else (
                "There's no specific information available on this, so answer from general knowledge."
            )
            t_encode0 = time.perf_counter()
            ids_before_trim = self.tokenizer.encode(ref_content)
            ids = ids_before_trim
            if len(ids) > self.max_ref_tokens:
                ids = ids[: self.max_ref_tokens]
                ref_content = self.tokenizer.decode(ids)
            new_ref_tokens = self.tokenizer.encode(search_helpers.wrap_with_ref_tags(ref_content))
            ref_encode_elapsed = time.perf_counter() - t_encode0
            self._turn_timing_stages["ref_encode"] = ref_encode_elapsed
            self.conv_logger.latency.stage(
                my_epoch, "ref_encode", ref_encode_elapsed,
                note=f"{len(new_ref_tokens)} token(s) ready to inject",
            )
            self.conv_logger.latency.count(
                my_epoch,
                grounding_tokens_before_trim=len(ids_before_trim),
                grounding_tokens_injected=len(new_ref_tokens),
                grounding_tokens_trimmed=max(0, len(ids_before_trim) - len(ids)),
                max_ref_tokens=int(self.max_ref_tokens),
            )
            self._pending_ref_token_counts = (len(ids_before_trim), self.max_ref_tokens)
            self.pending_ref_tokens = new_ref_tokens
            print(
                f"[liveTryPlasticity][search] prepared <ref> block "
                f"({len(self.pending_ref_tokens)} tok): {ref_content[:150]!r}",
                flush=True,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[liveTryPlasticity][search] route/search/compress failed: {e!r}\n{tb}", flush=True)
            self.conv_logger.error("route_and_search", e, tb)
            self.conv_logger.narrate_no_information(my_epoch)
            self.conv_logger.latency.mark(my_epoch, "grounding_ready", f"failed: {e!r}")
            if my_epoch == self.search_turn_epoch:
                fallback_tokens = self.tokenizer.encode(search_helpers.wrap_with_ref_tags(
                    "There's no specific information available on this, so answer from general knowledge."
                ))
                self._pending_ref_token_counts = (len(fallback_tokens), self.max_ref_tokens)
                self.pending_ref_tokens = fallback_tokens

    def _open_turn_latency(self, turn_id: int, transcript: str) -> None:
        """Open this turn's record in latency_<session>.log and seed it with
        the stages already measured in _stt_step. Never raises."""
        try:
            self._latency_turn_id = turn_id
            self._turn_text_token_count = 0
            self._turn_first_audio_generated_perf = None
            self._turn_audio_stream_started = False
            self._turn_last_audio_out_perf = None
            self._turn_audio_packets_out = 0
            self.conv_logger.latency.start_turn(
                turn_id,
                t0=self._turn_speech_end_perf if self._turn_speech_end_perf is not None else time.perf_counter(),
                transcript=transcript,
            )
            for name, secs in (self._pending_stt_stages or {}).items():
                self.conv_logger.latency.stage(turn_id, name, secs)
            if self._pending_stt_counts:
                self.conv_logger.latency.count(turn_id, **self._pending_stt_counts)
            self._pending_stt_stages = {}
            self._pending_stt_counts = {}
        except Exception:
            pass

    def _finish_turn_latency(self, response: str, outcome: str = "") -> None:
        """Close the turn currently being timed, once its spoken answer is known
        to be complete (the user has started speaking again). Never raises."""
        try:
            turn_id = self._latency_turn_id
            if turn_id is None:
                return
            end_perf = self._answer_end_perf
            if end_perf is None:
                rec = self.conv_logger.latency.get(turn_id)
                if rec is not None and self._last_text_emit_perf is not None:
                    if self._last_text_emit_perf >= rec.t0:
                        end_perf = self._last_text_emit_perf
            if end_perf is not None:
                self.conv_logger.latency.mark(
                    turn_id, "answer_complete", "last text token of the reply",
                    at=end_perf,
                )
            # Audio delivery is closed out SEPARATELY from answer_complete.
            # answer_complete above only says the model emitted its last text
            # token; these two say whether any audio reached the user at all.
            last_out = self._turn_last_audio_out_perf
            if last_out is not None:
                self.conv_logger.latency.mark(
                    turn_id, "audio_stream_completed",
                    "last audio packet written to the websocket", at=last_out,
                )
            self.conv_logger.latency.count(
                turn_id,
                answer_chars=len(response),
                answer_words=len(response.split()),
                answer_text_tokens=self._turn_text_token_count,
                audio_packets_streamed=self._turn_audio_packets_out,
                # The explicit "text finished but nothing was ever heard"
                # signal the previous logs could not distinguish.
                audio_delivered=bool(self._turn_audio_packets_out),
            )
            if not self._turn_audio_packets_out:
                self.conv_logger.event(
                    "audio_never_delivered",
                    "answer text completed but NO audio packet was ever sent for this turn",
                    turn=turn_id,
                )
            self.conv_logger.latency.finish_turn(turn_id, response=response, outcome=outcome)
            self._latency_turn_id = None
            self._answer_end_perf = None
        except Exception:
            pass

    def _log_timing_summary(self) -> None:
        """Print the consolidated big-to-small timing breakdown for the turn
        that was just committed (real ref, fallback, or router cancel -- all
        call this). Never raises."""
        try:
            start = getattr(self, "_turn_timing_start", None)
            stages = getattr(self, "_turn_timing_stages", None)
            if start is None or not stages:
                return
            total = time.perf_counter() - start
            self.conv_logger.turn_timing_summary(self.search_turn_epoch, total, dict(stages))
            self.conv_logger.narrate_timing_summary(self.search_turn_epoch, total, dict(stages))
        except Exception:
            pass

    def _finish_ref_injection(self, kind: str, elapsed: float) -> None:
        """Common bookkeeping once a <ref>/fallback injection's tokens have
        all been fed (whether that happened in one _consume_pending() call or
        was spread across several -- see _injection_in_progress).

        Releases suppress_text_until_ref HERE, at the end, not when injection
        starts: while _injection_in_progress is being drained across several
        ticks, _step()'s own per-tick call to self.lm_gen._step(codes...)
        still runs once per tick after _consume_pending() returns. If
        suppression were already released, that per-tick call would sample
        the model's OWN (real, unsuppressed) text -- interleaving it with the
        still-in-flight forced <ref> tokens instead of letting the model
        continue cleanly from the completed injection."""
        self.suppress_text_until_ref = False
        self._suppress_started_perf = None
        # -- Keep the outgoing audio masked while the injected text's own
        # audio drains out of the model. ------------------------------------
        # Forensic finding (logs_2/conversation.log): the assistant was
        # SPEAKING THE INJECTED REFERENCE ALOUD, tag fragments and all --
        # turn 6 said ". Class ( stock $. reflecting. move opened> Google
        # Class C Capital Stock price today is $339.52." and turn 5 said
        # ". As02, for $1ref> ...". _inject_tokens forces each <ref> token
        # into the text stream with moshi_tokens pinned to the zero frame, so
        # the injected STEPS themselves are silent -- but PersonaPlex's audio
        # codebooks lag its text stream, so the acoustic realisation of those
        # forced words comes out over the FOLLOWING normal steps and was
        # being decoded and streamed to the user. Masking roughly one step
        # per injected token (plus margin) covers exactly that drain window,
        # after which the model's real, grounded answer begins.
        #
        # The leak observed in logs_2 is the TAIL of the ref (8-10 words), not
        # the whole thing -- i.e. it is the fixed acoustic lag of the codec, so
        # the mask is a fixed WALL-CLOCK window (--ref_audio_drain_sec), not a
        # per-token count. A per-token count would mask for tens of seconds on
        # a long ref and swallow the real answer with it.
        drain_steps = max(0, int(round(self.ref_audio_drain_sec * (TARGET_SR / MIMI_FRAME_SIZE))))
        self._ref_audio_drain_remaining = drain_steps
        self.conv_logger.latency.count(
            self._latency_turn_id,
            ref_audio_drain_steps=drain_steps,
            ref_audio_drain_sec=round(self.ref_audio_drain_sec, 2),
        )
        n_tokens = self._injection_total_tokens
        token_text = self._injection_token_text
        self._turn_timing_stages["ref_inject"] = elapsed
        self.conv_logger.latency.stage(
            self._latency_turn_id, "ref_inject", elapsed,
            note=f"{n_tokens} token(s) fed into the live context ({kind})",
        )
        if kind == "ref":
            self.conv_logger.latency.mark(
                self._latency_turn_id, "ref_injected",
                f"after {self.search_filler_frame_count} filler chunk(s)",
            )
            self.conv_logger.ref_injected(token_text, n_tokens, elapsed, kind="ref")
            n_before, max_tok = self._pending_ref_token_counts
            self.conv_logger.narrate_injection(
                self.search_turn_epoch, token_text, n_tokens, n_before, max_tok, kind="ref",
            )
            self.conv_logger.turn_done(
                self.search_turn_epoch, "grounded from web search",
                time.perf_counter() - (self._turn_timing_start or time.perf_counter()),
            )
            print(
                f"[liveTryPlasticity][search] <ref> injected ({n_tokens} tok) "
                f"after {self.search_filler_frame_count} filler chunks",
                flush=True,
            )
        else:  # fallback (filler timeout)
            self.conv_logger.latency.mark(
                self._latency_turn_id, "ref_injected",
                f"filler timeout after {self.search_filler_frame_count} chunk(s) -- "
                f"the search did not finish in time",
            )
            self.conv_logger.latency.count(
                self._latency_turn_id, search_timed_out=True, grounding_tokens_injected=n_tokens,
            )
            self.conv_logger.ref_injected(token_text, n_tokens, elapsed, kind="ref_fallback")
            self.conv_logger.narrate_injection(
                self.search_turn_epoch, token_text, n_tokens, n_tokens, self.max_ref_tokens, kind="ref_fallback",
            )
            self.conv_logger.turn_done(
                self.search_turn_epoch, "search timed out, answered from own knowledge",
                time.perf_counter() - (self._turn_timing_start or time.perf_counter()),
            )
            print("[liveTryPlasticity][search] <ref> fallback injected after filler timeout", flush=True)
        self._log_timing_summary()
        # Start the stuck-silence watchdog: real speech should resume within
        # post_inject_watchdog_sec. See _step()'s post-injection check.
        self._post_inject_watch_started = time.perf_counter()
        self._post_inject_watch_turn = self.search_turn_epoch
        self._post_inject_watch_fired = False

    def _consume_pending(self) -> None:
        """Called once per chunk while a routing/search decision is in flight.
        This is the ONLY place the background thread's work reaches the LM,
        because token injection must happen on the GPU thread.

        Checked in priority order:
          0. an injection already in progress -- drain up to
             inject_tokens_per_tick more tokens of it and return. Spreading a
             20-30 token <ref> injection across several ticks (instead of
             blocking one _step() call for up to ~1.3s, as measured in
             logs_1/conversation.log turn 7) keeps mic ingestion and avatar
             rendering close to their normal cadence during it.
          1. cancelled  -- the router decided no search is needed; stop the
                           thinking sound and let the model answer normally.
          2. ref ready  -- start injecting the grounded <ref> block.
          3. timed out  -- after self._SEARCH_MAX_FILLER_FRAMES chunks start
                           injecting a generic fallback so the model never
                           hangs waiting on a slow or failed search.
        Independently of all of the above: if suppression has lasted longer
        than max_suppress_sec, release it early so the model is not held
        artificially silent indefinitely -- the in-flight search keeps
        running and its <ref> (or the timeout fallback) still gets injected
        normally once ready."""
        import search_helpers

        self.search_filler_frame_count += 1

        if self._injection_in_progress is not None:
            kind, remaining = self._injection_in_progress
            batch = remaining[: self.inject_tokens_per_tick]
            self._inject_tokens(batch)
            remaining = remaining[self.inject_tokens_per_tick:]
            if remaining:
                self._injection_in_progress = (kind, remaining)
                return
            self._injection_in_progress = None
            elapsed = time.perf_counter() - self._injection_started_perf
            self._finish_ref_injection(kind, elapsed)
            return

        # The cap must NOT fire once a grounding is already in hand or being
        # fed. logs_4 turn 3 shows exactly that failure: "suppress_cap_released
        # waited_s=3.1" landed while ref_inject was mid-flight (that turn's
        # injection ran +3.14s -> +5.58s), so forced silence lifted while
        # forced <ref> tokens were still arriving. The model then sampled its
        # OWN text interleaved with the remaining injected tokens -- the same
        # real/forced interleaving that moving the release into
        # _finish_ref_injection was meant to prevent. The cap exists to stop
        # the model hanging on a SLOW SEARCH; once the answer is here, or
        # already going in, there is nothing left to wait for.
        if (
            self.suppress_text_until_ref
            and self._suppress_started_perf is not None
            and self._injection_in_progress is None
            and self.pending_ref_tokens is None
            and (time.perf_counter() - self._suppress_started_perf) > self.max_suppress_sec
        ):
            self.suppress_text_until_ref = False
            self.conv_logger.event(
                "suppress_cap_released", turn=self.search_turn_epoch,
                waited_s=round(time.perf_counter() - self._suppress_started_perf, 2),
            )
            print(
                f"[liveTryPlasticity][search] max_suppress_sec ({self.max_suppress_sec:.1f}s) reached -- "
                f"releasing forced silence early; the search keeps running in the background",
                flush=True,
            )

        if self.pending_search_cancelled:
            self.pending_search_cancelled = False
            self.pending_lookup_tokens = None
            self.pending_start_thinking = False
            self.suppress_text_until_ref = False
            self._suppress_started_perf = None
            self.search_awaiting_ref = False
            self.search_ref_committed_this_turn = True
            self._stop_thinking_sound(
                self.search_turn_epoch, "no_search_needed",
                "the assistant already knew this and did not need to search",
            )
            self.conv_logger.latency.count(
                self._latency_turn_id, injected_anything=False,
            )
            print(
                "[liveTryPlasticity][search] router said no search -- answering from "
                "the model's own knowledge (nothing injected)",
                flush=True,
            )
            return

        if self.pending_start_thinking:
            self.pending_start_thinking = False
            if self.pending_ref_tokens is None:
                self._start_thinking_sound(self.search_turn_epoch, self.search_current_transcript)

        if self.pending_lookup_tokens is not None:
            lookup_tokens = self.pending_lookup_tokens
            self.pending_lookup_tokens = None
            t_lookup0 = time.perf_counter()
            self._inject_tokens(lookup_tokens)
            lookup_elapsed = time.perf_counter() - t_lookup0
            self._turn_timing_stages["lookup_inject"] = lookup_elapsed
            self.conv_logger.latency.stage(
                self._latency_turn_id, "lookup_inject", lookup_elapsed,
                note=f"{len(lookup_tokens)} token(s)",
            )
            self.conv_logger.latency.mark(self._latency_turn_id, "lookup_injected")
            self.conv_logger.ref_injected(
                self.tokenizer.decode(lookup_tokens), len(lookup_tokens), lookup_elapsed, kind="lookup"
            )
            print(
                f"[liveTryPlasticity][search] <lookup> injected ({len(lookup_tokens)} tok) "
                f"-- searching in background",
                flush=True,
            )

        if self.pending_ref_tokens is not None:
            ref_tokens = self.pending_ref_tokens
            self.pending_ref_tokens = None
            self.search_awaiting_ref = False
            self.search_ref_committed_this_turn = True
            # Force ON (not merely leave as-is): max_suppress_sec may have
            # already released it while we were still waiting for this ref to
            # arrive. It must be True for the full (possibly multi-tick)
            # duration of the injection -- released in _finish_ref_injection,
            # not here. See that method's docstring for why.
            self.suppress_text_until_ref = True
            self._defer_thinking_sound_stop(
                self.search_turn_epoch, "ref_ready", "the answer was ready",
            )
            self._injection_started_perf = time.perf_counter()
            self._injection_total_tokens = len(ref_tokens)
            self._injection_token_text = self.tokenizer.decode(ref_tokens)
            first_batch = ref_tokens[: self.inject_tokens_per_tick]
            self._inject_tokens(first_batch)
            rest = ref_tokens[self.inject_tokens_per_tick:]
            if rest:
                self._injection_in_progress = ("ref", rest)
            else:
                self._finish_ref_injection("ref", time.perf_counter() - self._injection_started_perf)
        elif self.search_filler_frame_count >= self._SEARCH_MAX_FILLER_FRAMES:
            fallback_text = "There's no specific information available on this, so answer from general knowledge."
            fallback = self.tokenizer.encode(search_helpers.wrap_with_ref_tags(fallback_text))
            self.search_awaiting_ref = False
            self.search_ref_committed_this_turn = True
            # Force ON for the same reason as the ref-ready branch above --
            # released in _finish_ref_injection, not here.
            self.suppress_text_until_ref = True
            self._defer_thinking_sound_stop(
                self.search_turn_epoch, "filler_timeout",
                "the search was taking too long, so the assistant moved on with what it had",
            )
            self._injection_started_perf = time.perf_counter()
            self._injection_total_tokens = len(fallback)
            self._injection_token_text = fallback_text
            first_batch = fallback[: self.inject_tokens_per_tick]
            self._inject_tokens(first_batch)
            rest = fallback[self.inject_tokens_per_tick:]
            if rest:
                self._injection_in_progress = ("fallback", rest)
            else:
                self._finish_ref_injection("fallback", time.perf_counter() - self._injection_started_perf)

    @torch.no_grad()
    def _step(self, pcm24: np.ndarray) -> dict:
        self.step += 1
        t0 = time.perf_counter()
        chunk = torch.from_numpy(pcm24).to(self.device, dtype=torch.float32)[None, None]

        t_encode0 = time.perf_counter()
        codes = self.mimi.encode(chunk)
        t_encode1 = time.perf_counter()
        if self.skip_first:
            self.mimi.reset_streaming()
            self.skip_first = False

        # -- STT/VAD forward pass + turn-boundary detection, then consume any
        # in-flight routing/search result. No-op (both guards false) unless
        # --stt_hf_repo/--stt_pkg_dir were configured at launch, so the plain
        # conversational path is untouched.
        #
        # Both calls run inside try/except: this hook runs on every single
        # chunk of the live conversation, so an uncaught exception here must
        # not propagate out of _step() and kill the entire GPU producer
        # thread. On failure, search is disabled for the rest of this session
        # but the avatar keeps talking.
        if self.stt_lm_gen is not None and not self.search_hard_disabled:
            try:
                self._stt_step(chunk)
            except Exception as e:
                tb = traceback.format_exc()
                print(
                    f"[liveTryPlasticity][search] _stt_step failed, disabling search "
                    f"for the rest of this session: {e!r}\n{tb}",
                    flush=True,
                )
                self.conv_logger.error("stt_step", e, tb)
                self.search_hard_disabled = True
                self.search_awaiting_ref = False
                self._injection_in_progress = None
                # Through _stop_thinking_sound (not a bare flag assignment) so
                # the stop is still logged with its duration on this path.
                self._stop_thinking_sound(
                    self.search_turn_epoch, "pipeline_error",
                    "the search pipeline failed, so the assistant answered without it",
                )
                self._thinking_stop_pending = None
                self._ref_audio_drain_remaining = 0
                self.suppress_text_until_ref = False
        # Also keep polling while a ref/fallback injection is being drained
        # across multiple ticks (self._injection_in_progress) -- that flag
        # stays set for a few ticks AFTER search_awaiting_ref has already
        # gone False (it is cleared the instant injection starts, not when it
        # finishes), so consume_pending must still be called until it empties.
        if (
            (self.search_awaiting_ref or self._injection_in_progress is not None)
            and not self.search_hard_disabled
        ):
            try:
                self._consume_pending()
            except Exception as e:
                tb = traceback.format_exc()
                print(
                    f"[liveTryPlasticity][search] _consume_pending failed, disabling "
                    f"search for the rest of this session: {e!r}\n{tb}",
                    flush=True,
                )
                self.conv_logger.error("consume_pending", e, tb)
                self.search_hard_disabled = True
                self.search_awaiting_ref = False
                self._injection_in_progress = None
                # Through _stop_thinking_sound (not a bare flag assignment) so
                # the stop is still logged with its duration on this path.
                self._stop_thinking_sound(
                    self.search_turn_epoch, "pipeline_error",
                    "the search pipeline failed, so the assistant answered without it",
                )
                self._thinking_stop_pending = None
                self._ref_audio_drain_remaining = 0
                self.suppress_text_until_ref = False

        # While a search is in flight, hand the model its own "say nothing"
        # token instead of letting it sample text -- the model's native
        # silence mechanism. Audio and hidden states keep flowing (so the
        # avatar pipeline and chunk cadence are untouched); only the words are
        # withheld, precisely while the model would otherwise be inventing an
        # answer it is about to be handed.
        t_lm0 = time.perf_counter()
        if self.suppress_text_until_ref and self._step_supports_text_token:
            lm_out = self.lm_gen._step(
                codes[:, :, :1], text_token=getattr(self.lm_gen, "zero_text_code", 3)
            )
        else:
            lm_out = self.lm_gen._step(codes[:, :, :1])
        t_lm1 = time.perf_counter()

        tokens = None
        helium_hidden = None
        if lm_out is not None:
            if not (isinstance(lm_out, tuple) and len(lm_out) == 3):
                raise RuntimeError(f"Moshi graph layer[-2] contract failure: got {type(lm_out)} len={len(lm_out) if isinstance(lm_out, tuple) else 'n/a'}")
            tokens, _transformer_out, layer_hidden = lm_out
            helium_hidden = layer_hidden[:1, -1:].detach().float().cpu()

        token = -1
        token_piece = ""
        decode_ms = 0.0
        reply_codes = None
        if tokens is None:
            reply_pcm = np.zeros(MIMI_FRAME_SIZE, dtype=np.float32)
        else:
            token = int(tokens[0, 0, 0].detach().item())
            token_piece = self.decode_piece(token)
            if token_piece:
                self.audio_text += token_piece
                # Running end-of-answer marker for the latency log: the last
                # text token the model emitted. Frozen in _stt_step the moment
                # the user starts speaking again (see _answer_end_perf).
                self._last_text_emit_perf = time.perf_counter()
                self._turn_text_token_count += 1
            reply_codes = tokens[:, 1:].detach().to(device="cpu", dtype=torch.int16)
            t_decode0 = time.perf_counter()
            reply = self.mimi.decode(tokens[:, 1:])
            reply_pcm = reply[0, 0].detach().float().cpu().numpy()
            decode_ms = 1000.0 * (time.perf_counter() - t_decode0)
            if reply_pcm.shape[0] < MIMI_FRAME_SIZE:
                reply_pcm = np.pad(reply_pcm, (0, MIMI_FRAME_SIZE - reply_pcm.shape[0]))
            elif reply_pcm.shape[0] > MIMI_FRAME_SIZE:
                reply_pcm = reply_pcm[:MIMI_FRAME_SIZE]

        # The model's OWN output level, measured before the thinking sound can
        # replace it below: after the swap, `reply_pcm` may be the filler clip,
        # whose level says nothing about whether the model is speaking.
        model_own_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))

        # -- Post-injection stuck-silence watchdog ---------------------------
        # Forensic finding (logs_1/conversation.log, RunPod RTX 5090,
        # 2026-09-05, turn 7): a clean, on-time <ref> injection -- no timeout,
        # no fallback -- was followed by 50+ seconds of the model producing
        # ONLY padding/silence text tokens, discovered only because the NEXT
        # question's VAD eventually fired. That is far too slow to notice or
        # diagnose. This does not attempt to fix the model's behavior (there
        # is no evidence-backed intervention available from this side of the
        # API); it detects the same condition within post_inject_watchdog_sec
        # and closes the turn's latency record out immediately, so "answer
        # complete" is never confused with "audio was actually delivered".
        if self._post_inject_watch_started is not None:
            if token_piece or model_own_rms > self._SPEECH_RMS_THRESHOLD:
                self._post_inject_watch_started = None
                self._post_inject_watch_fired = False
            elif (
                not self._post_inject_watch_fired
                and (time.perf_counter() - self._post_inject_watch_started) > self.post_inject_watchdog_sec
            ):
                self._post_inject_watch_fired = True
                stuck_elapsed = time.perf_counter() - self._post_inject_watch_started
                watch_turn = self._post_inject_watch_turn
                runtime_logging.log_event(
                    runtime_logging.get_system_logger(), "PersonaPlex", "post_injection_silence",
                    level=logging.ERROR, turn=watch_turn, elapsed_s=round(stuck_elapsed, 2),
                )
                self.conv_logger.event(
                    "post_injection_silence_watchdog",
                    f"turn={watch_turn} produced no text/audio {stuck_elapsed:.1f}s after the "
                    f"answer was injected -- treating audio delivery as failed for this turn",
                    turn=watch_turn, elapsed_s=round(stuck_elapsed, 2),
                )
                self.conv_logger.latency.count(
                    self._latency_turn_id, audio_delivery_status="stuck_no_audio_after_injection",
                )
                self._finish_turn_latency(
                    "", outcome=f"stuck silence: no audio {stuck_elapsed:.1f}s after injection",
                )

        # First real audio of this turn -> the honest end-to-end latency.
        was_awaiting_first_speech = self._turn_awaiting_first_speech
        if self._turn_awaiting_first_speech and model_own_rms > self._SPEECH_RMS_THRESHOLD:
            self._turn_awaiting_first_speech = False
            started = self._turn_timing_start
            if started is not None:
                backlog_s = float(self.input_buffer.shape[0]) / TARGET_SR
                self.conv_logger.turn_spoke(
                    self._turn_first_speech_epoch,
                    time.perf_counter() - started,
                    backlog_s,
                )
                self.conv_logger.latency.mark(
                    self._turn_first_speech_epoch, "first_word",
                    f"microphone backlog {backlog_s:.2f}s at this moment",
                )
                # Distinct from first_word (a text/turn concept) and from
                # answer_complete (text only): this is the instant the MODEL
                # produced audible PCM. note_audio_streamed() later records
                # when that audio actually left the websocket; the difference
                # is the avatar/chunking/prebuffer delay, which nothing in the
                # pipeline measured before.
                self._turn_first_audio_generated_perf = time.perf_counter()
                self.conv_logger.latency.mark(
                    self._turn_first_speech_epoch, "first_audio_generated",
                    "model emitted its first non-silent frame (not yet sent)",
                )
                self.conv_logger.latency.count(
                    self._turn_first_speech_epoch,
                    input_backlog_s_at_first_word=round(backlog_s, 3),
                )

        # "Thinking sound": while an online search is in flight, replace what
        # the model would otherwise output with the looped clip. The model
        # keeps stepping normally above (KV cache / timing untouched); only the
        # audio actually sent out is swapped. force_idle tells the avatar-gate
        # downstream to stay visually idle rather than lip-syncing to this
        # non-speech sound (its RMS is well above the speech threshold, so
        # without this the avatar would appear to "talk").
        #
        # The sound also covers the ref-audio DRAIN window: after the injected
        # tokens are fed, PersonaPlex's audio codebooks are still emitting the
        # reference text as speech (logs_2). _defer_thinking_sound_stop() armed
        # the stop instead of taking it, and _finish_ref_injection() set the
        # drain budget; the stop only lands once that budget is spent, i.e.
        # once the model is genuinely about to speak its own answer.
        if self._ref_audio_drain_remaining > 0:
            self._ref_audio_drain_remaining -= 1
            if self._ref_audio_drain_remaining == 0 and self._thinking_stop_pending is not None:
                reason, reason_text, stop_turn_id = self._thinking_stop_pending
                self._stop_thinking_sound(stop_turn_id, reason, reason_text)
        force_idle = False
        if self.search_thinking_active and self.thinking_sound_pcm is not None:
            reply_pcm = self._next_thinking_sound_chunk()
            force_idle = True
        elif self._ref_audio_drain_remaining > 0:
            # No clip configured to cover the drain -- emit silence rather than
            # letting the leaked reference audio reach the user.
            reply_pcm = np.zeros_like(reply_pcm)
            force_idle = True

        reply_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))
        reply_peak = float(np.max(np.abs(reply_pcm))) if reply_pcm.size else 0.0
        input_rms = float(np.sqrt(np.mean(np.square(pcm24, dtype=np.float32))))
        encode_ms = 1000.0 * (t_encode1 - t_encode0)
        lm_ms = 1000.0 * (t_lm1 - t_lm0)
        total_ms = 1000.0 * (time.perf_counter() - t0)

        # Charge this 80ms frame's GPU cost to the turn being timed, once per
        # turn's worth of frames the user actually cares about (waiting for
        # the first word, or the model audibly speaking).
        if self._latency_turn_id is not None and (
            was_awaiting_first_speech or model_own_rms > self._SPEECH_RMS_THRESHOLD
        ):
            self.conv_logger.latency.accumulate(self._latency_turn_id, "gen_lm", (t_lm1 - t_lm0))
            self.conv_logger.latency.accumulate(self._latency_turn_id, "gen_audio_decode", decode_ms / 1000.0)
            self.conv_logger.latency.accumulate(self._latency_turn_id, "gen_audio_encode", (t_encode1 - t_encode0))

        reply_i16 = np.clip(reply_pcm, -1.0, 1.0)
        reply_i16 = (reply_i16 * 32767.0).astype(np.int16)
        audio_b64 = base64.b64encode(reply_i16.tobytes()).decode("ascii")

        print(
            "[liveTryStudio] moshi "
            f"step={self.step} token={token} piece={token_piece!r} "
            f"in_rms={input_rms:.5f} reply_rms={reply_rms:.5f} peak={reply_peak:.3f} "
            f"hidden={helium_hidden is not None} "
            f"encode={encode_ms:.1f}ms lm={lm_ms:.1f}ms decode={decode_ms:.1f}ms total={total_ms:.1f}ms",
            flush=True,
        )

        return {
            "step": int(self.step),
            "sample_rate": TARGET_SR,
            "reply_i16_b64": audio_b64,
            "reply_rms": reply_rms,
            "reply_peak": reply_peak,
            "input_rms": input_rms,
            "token": token,
            "piece": token_piece,
            "sampled_text": self.sampled_text,
            "audio_text": self.audio_text,
            "encode_ms": encode_ms,
            "lm_ms": lm_ms,
            "decode_ms": decode_ms,
            "total_ms": total_ms,
            "helium_hidden": helium_hidden,
            "reply_codes": reply_codes,
            "force_idle": force_idle,
        }

    @torch.no_grad()
    def process_ready_steps_limited(self, max_steps: int) -> list[dict]:
        """Process a bounded number of Mimi frames.

        The base MoshiOnlyEngine drains the entire input buffer before returning.
        In live mode that is dangerous: mic audio can accumulate while Moshi is
        loading, then avatar frames do not reach the sender until the backlog is
        fully processed. Bounded draining keeps the producer/sender interleaved.
        """
        events: list[dict] = []
        for _ in range(max(1, int(max_steps))):
            if self.input_buffer.shape[0] < MIMI_FRAME_SIZE:
                break
            pcm = self.input_buffer[:MIMI_FRAME_SIZE].copy()
            self.input_buffer = self.input_buffer[MIMI_FRAME_SIZE:].copy()
            events.append(self._step(pcm))
        return events


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _ms(t0: float) -> float:
    return 1000.0 * (time.perf_counter() - t0)


def encode_jpeg_b64(frame_rgb: np.ndarray, quality: int) -> str:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return base64.b64encode(enc.tobytes()).decode("ascii")


def encode_jpeg_bytes(frame_rgb: np.ndarray, quality: int) -> bytes:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return enc.tobytes()


def _pcm_f32_to_i16_b64(pcm: np.ndarray) -> str:
    arr = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return base64.b64encode((arr * 32767.0).astype(np.int16).tobytes()).decode("ascii")


def _pcm_f32_to_i16_bytes(pcm: np.ndarray) -> bytes:
    arr = np.clip(np.asarray(pcm, dtype=np.float32), -1.0, 1.0)
    return (arr * 32767.0).astype(np.int16).tobytes()


def split_audio_into_frame_slices(pcm: np.ndarray, fps: float) -> list[np.ndarray]:
    frame_samples = int(round(TARGET_SR / float(fps)))
    arr = np.asarray(pcm, dtype=np.float32)
    n_frames = max(0, int(round(arr.shape[0] / frame_samples)))
    if n_frames == 0:
        return []
    total = n_frames * frame_samples
    if arr.shape[0] < total:
        arr = np.pad(arr, (0, total - arr.shape[0]))
    elif arr.shape[0] > total:
        arr = arr[:total]
    return [arr[i * frame_samples:(i + 1) * frame_samples].copy() for i in range(n_frames)]


def load_audio_24k(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0).float().numpy()


def load_ref_image(path: str | Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((512, 512), Image.LANCZOS)
    return T.ToTensor()(img).unsqueeze(0).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# FM + Renderer weight loading (identical to liveTryFM.py)
# ---------------------------------------------------------------------------

def _clean_generator_state(ckpt: dict) -> dict:
    raw = ckpt.get("ema_state_dict") or ckpt.get("state_dict", ckpt.get("model", ckpt))
    if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    return {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in raw.items()}


def _load_fm(args: argparse.Namespace, device: torch.device) -> FMGenerator:
    syslog = runtime_logging.get_system_logger()
    t_total = time.perf_counter()
    with runtime_logging.Timer(
        syslog, "IMTalker.FMGenerator", "load_base",
        path=args.generator_path, device=str(device),
    ):
        fm = FMGenerator(args).to(device).eval()
        ckpt = torch.load(args.generator_path, map_location="cpu")
        cleaned = _clean_generator_state(ckpt)
        missing, unexpected = fm.load_state_dict(cleaned, strict=False)
    print(
        f"[liveTryHeliumFM][FM] base loaded missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    lora_path = str(getattr(args, "lora_generator_path", "") or "")
    if lora_path:
        with runtime_logging.Timer(
            syslog, "IMTalker.AvatarMotionLoRA", "load",
            name="ditto_blink_lora", path=lora_path, base_model="FMGenerator",
            device=str(device), rank=int(getattr(args, "lora_rank", 64) or 64),
        ):
            apply_lora_to_model(
                fm,
                rank=int(getattr(args, "lora_rank", 64) or 64),
                alpha=float(getattr(args, "lora_alpha", 128) or 128),
                dropout=float(getattr(args, "lora_dropout", 0.05)),
                include_pose_lora=not bool(getattr(args, "no_lora_pose_projection", False)),
                include_audio_lora=not bool(getattr(args, "no_lora_audio_projection", False)),
                only_pose_lora=bool(getattr(args, "only_lora_pose_projection", False)),
            )
            lora_ckpt = torch.load(lora_path, map_location="cpu")
            lora_cleaned = _clean_generator_state(lora_ckpt)
            missing_lora, unexpected_lora = fm.load_state_dict(lora_cleaned, strict=False)
            lora_keys = sum(1 for key in lora_cleaned if "lora_" in key)
        print(
            f"[liveTryHeliumFM][FM] lora loaded path={lora_path} "
            f"lora_keys={lora_keys} missing={len(missing_lora)} unexpected={len(unexpected_lora)}",
            flush=True,
        )
    else:
        runtime_logging.log_event(syslog, "IMTalker.AvatarMotionLoRA", "not_configured")
    fm.to(device).eval()
    _sync_cuda()
    print(f"[liveTryHeliumFM][FM] loaded in {_ms(t_total):.0f}ms", flush=True)
    return fm


def _load_renderer(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> IMTRenderer:
    syslog = runtime_logging.get_system_logger()
    t_total = time.perf_counter()
    with runtime_logging.Timer(
        syslog, "IMTalker.Renderer", "load",
        path=args.renderer_path, device=str(device), dtype=str(dtype),
    ):
        renderer = IMTRenderer(args).to(device).eval()
        ckpt = torch.load(args.renderer_path, map_location="cpu")
        raw = ckpt.get("state_dict", ckpt.get("model", ckpt))
        cleaned = {k.replace("gen.", "", 1).replace("model.", "", 1): v for k, v in raw.items()}
        missing, unexpected = renderer.load_state_dict(cleaned, strict=False)
        renderer = renderer.to(dtype=dtype)
        _sync_cuda()
        if getattr(args, "compile_renderer", False):
            @torch.no_grad()
            def _fused_render(motion_latent, g_r, m_r, f_r):
                ta_c = renderer.adapt(motion_latent, g_r)
                m_c = renderer.latent_token_decoder(ta_c)
                frames = renderer.decode(m_c, m_r, f_r)
                return frames
            renderer._fused_render = torch.compile(_fused_render)
    print(
        f"[liveTryHeliumFM][renderer] loaded in {_ms(t_total):.0f}ms "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return renderer


# ---------------------------------------------------------------------------
# Helium extractor (chunk-local, batch LM)
# ---------------------------------------------------------------------------

class HeliumExtractor:
    """Prefix-growing raw Helium + global interpolation.

    This matches the best Stage 3 diagnostic:
      audio prefix -> raw Helium -> append only new raw steps
      -> one global interpolation over accumulated raw Helium
      -> emit last target_frames
    """

    def __init__(
        self,
        helium_mimi: "MimiModel",
        helium_lm: "LMModel",
        device: torch.device,
    ) -> None:
        self.helium_mimi = helium_mimi
        self.helium_lm = helium_lm
        self.device = device
        self._lock = threading.Lock()
        self._prefix_pcm = np.empty(0, dtype=np.float32)
        self._raw_parts: list[torch.Tensor] = []
        self._prev_raw_len = 0
        self._emitted_frames = 0

    def reset(self) -> None:
        with self._lock:
            self._prefix_pcm = np.empty(0, dtype=np.float32)
            self._raw_parts = []
            self._prev_raw_len = 0
            self._emitted_frames = 0

    def _extract_raw(self, pcm_np: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(np.asarray(pcm_np, dtype=np.float32)).to(self.device, dtype=torch.float32)[None, None]

        codes = self.helium_mimi.encode(wav)
        codes = codes[:, :MAIN_CODEBOOKS, :].detach()
        batch_size, n_q, total_steps = codes.shape

        dtype = next(self.helium_lm.parameters()).dtype
        input_emb = torch.zeros(
            batch_size, total_steps, self.helium_lm.dim, device=self.device, dtype=dtype
        )
        for q in range(n_q):
            input_emb = input_emb + self.helium_lm.emb[q](codes[:, q].long())

        padding_ids = torch.full(
            (batch_size, total_steps),
            self.helium_lm.existing_text_padding_id,
            dtype=torch.long,
            device=self.device,
        )
        input_emb = input_emb + self.helium_lm.text_emb(padding_ids)

        if getattr(self.helium_lm.transformer, "_streaming_state", None) is not None:
            raise RuntimeError("helium_lm must stay in batch mode (non-streaming)")

        captured: list[torch.Tensor] = []

        def _hook(_mod, _inp, out):
            captured.append(out.detach())

        handle = self.helium_lm.transformer.layers[-2].register_forward_hook(_hook)
        try:
            self.helium_lm.transformer(input_emb)
        finally:
            if reply_engine is not None:
                pipeline_stop_event.set()
                session_started.clear()
            handle.remove()

        if len(captured) != 1:
            raise RuntimeError(f"Helium hook captured {len(captured)} tensors; expected 1")
        return captured[0].squeeze(0).float().contiguous()  # [T_raw, 4096]

    @torch.no_grad()
    def extract_raw_chunk(self, pcm_np: np.ndarray) -> torch.Tensor:
        """Return only the new raw 12.5Hz Helium steps for one new audio chunk."""
        pcm = np.asarray(pcm_np, dtype=np.float32)
        if pcm.ndim != 1 or pcm.size == 0:
            raise RuntimeError("HeliumExtractor.extract_raw_chunk expects non-empty 1D PCM")

        with self._lock:
            self._prefix_pcm = np.concatenate([self._prefix_pcm, pcm], axis=0)
            raw_prefix = self._extract_raw(self._prefix_pcm)
            new_raw = raw_prefix[self._prev_raw_len:]
            if int(new_raw.shape[0]) == 0 and int(raw_prefix.shape[0]) > 0:
                new_raw = raw_prefix[-1:]
            self._raw_parts.append(new_raw.cpu())
            self._prev_raw_len = int(raw_prefix.shape[0])
            return new_raw.contiguous()

    @torch.no_grad()
    def extract_exact_chunk_from_prefix(
        self,
        pcm_prefix: np.ndarray,
        chunk_start_frame: int,
        target_frames: int,
    ) -> torch.Tensor:
        """Return the raw 12.5Hz Helium slice for a video-frame window.

        This path is used by file-mode lookahead. It must return raw Helium
        steps so the studio adapter remains the only temporal upsampler.
        """
        pcm = np.asarray(pcm_prefix, dtype=np.float32)
        if pcm.ndim != 1 or pcm.size == 0:
            raise RuntimeError("HeliumExtractor.extract_exact_chunk_from_prefix expects non-empty 1D PCM")
        with self._lock:
            raw_prefix = self._extract_raw(pcm)
        start_frame = int(chunk_start_frame)
        end_frame = start_frame + int(target_frames)
        start_raw = int(round(start_frame * 0.5))
        end_raw = int(round(end_frame * 0.5))
        start_raw = max(0, min(start_raw, int(raw_prefix.shape[0])))
        end_raw = max(start_raw + 1, min(end_raw, int(raw_prefix.shape[0])))
        if end_raw > int(raw_prefix.shape[0]):
            raise RuntimeError(
                f"Requested raw slice [{start_raw}, {end_raw}) exceeds prefix Helium length {raw_prefix.shape[0]}"
            )
        return raw_prefix[start_raw:end_raw].contiguous()


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class LiveHeliumFMEngine:
    """Helium extraction + FM + renderer, session-stateful."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        _t_engine_init = time.perf_counter()
        _syslog = runtime_logging.get_system_logger()
        self.args = args
        self.device = torch.device(args.device)
        renderer_precision = str(
            getattr(args, "renderer_precision", "fp32")
        ).lower()
        self.dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[renderer_precision]
        self.fps = float(args.fps)
        self.audio_chunk_sec = float(getattr(args, "audio_chunk_sec", 0.96))
        self.audio_chunk_samples = int(round(self.audio_chunk_sec * TARGET_SR))
        self.fm_chunk_frames = max(1, int(getattr(args, "fm_chunk_frames", 24)))
        self.live_sliding_window = bool(getattr(args, "enable_live_sliding_window", False))
        self.slide_past_frames = max(0, int(getattr(args, "slide_past_frames", 10)))
        self.slide_future_frames = max(0, int(getattr(args, "slide_future_frames", 3)))
        self.render_sub_batch = max(1, int(args.render_sub_batch))
        self.jpeg_quality = int(args.jpeg_quality)
        trained_window = int(round(float(args.wav2vec_sec) * self.fps))
        if self.fm_chunk_frames != trained_window:
            print(
                f"[liveTryHeliumFM] WARNING fm_chunk_frames={self.fm_chunk_frames} "
                f"but wav2vec_sec*fps={trained_window}",
                flush=True,
            )
        if self.live_sliding_window:
            print(
                f"[liveTryHeliumFM][typeAC] IMTalker lookahead enabled "
                f"past={self.slide_past_frames}f current={self.fm_chunk_frames}f "
                f"future={self.slide_future_frames}f",
                flush=True,
            )

        t_total = time.perf_counter()

        # FM + renderer
        self.fm = _load_fm(args, self.device)
        self.renderer = _load_renderer(args, self.device, self.dtype)

        # Adapter: either the six-layer projected-frontend model or UniTalk's
        # 12-layer model with only its final layer passed to IMTalker.
        t_adapter = time.perf_counter()
        if args.adapter_type == "unitalk_last_layer":
            self.studio_adapter = UniTalkLastLayerLiveAdapter(
                args.wav2vec_model_path,
                args.adapter_dropout,
            ).to(self.device).float().eval()
        else:
            self.studio_adapter = StudioNativeLiveAdapter(
                args.wav2vec_model_path,
                args.adapter_num_layers,
                args.adapter_dropout,
            ).to(self.device).float().eval()
        payload = torch.load(args.adapter_path, map_location="cpu")
        if isinstance(payload, dict) and args.adapter_type == "frontend":
            saved_args = payload.get("args", {})
            if saved_args and int(saved_args.get("num_layers", args.adapter_num_layers)) != int(args.adapter_num_layers):
                print(
                    f"[liveTryHeliumFrontendFM] WARNING checkpoint num_layers={saved_args.get('num_layers')} "
                    f"but CLI adapter_num_layers={args.adapter_num_layers}",
                    flush=True,
                )
            state = payload.get("adapter", payload.get("model", payload))
        else:
            state = payload
        self.studio_adapter.load_state_dict(state, strict=True)
        _sync_cuda()
        print(
            f"[liveTryHeliumFrontendFM][adapter] type={args.adapter_type} loaded in {_ms(t_adapter):.0f}ms "
            f"path={args.adapter_path}",
            flush=True,
        )
        runtime_logging.log_event(
            _syslog, "IMTalker.StudioAdapter", "loaded",
            adapter_type=args.adapter_type, path=args.adapter_path,
            device=str(self.device), duration_ms=round(_ms(t_adapter), 1),
        )

        # Raw HF Wav2Vec2 target path used during Helium adapter training.
        t_w2v = time.perf_counter()
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.wav2vec_model_path,
            local_files_only=True,
        )
        self.wav2vec_model = self.studio_adapter.wav2vec
        _sync_cuda()
        print(
            f"[liveTryHeliumStudioFM][wav2vec] loaded in {_ms(t_w2v):.0f}ms "
            f"path={args.wav2vec_model_path}",
            flush=True,
        )
        runtime_logging.log_event(
            _syslog, "IMTalker.Wav2Vec2", "loaded",
            path=args.wav2vec_model_path, device=str(self.device),
            duration_ms=round(_ms(t_w2v), 1),
        )

        # Reference image: pre-compute identity + motion-ref features once
        ref_tensor = load_ref_image(args.ref_path, self.device, self.dtype)
        with torch.no_grad():
            self.f_r, self.g_r = self.renderer.dense_feature_encoder(ref_tensor)
            self.ref_x = self.renderer.latent_token_encoder(ref_tensor).to(dtype=torch.float32)
            ta_r = self.renderer.adapt(self.ref_x.to(dtype=self.dtype), self.g_r)
            self.m_r = self.renderer.latent_token_decoder(ta_r)
        _sync_cuda()
        self.eye_blink_enabled = bool(getattr(args, "enable_eye_blink_composite", False))
        self._blink_maps: tuple[torch.Tensor, ...] | None = None
        self._eye_masks: tuple[torch.Tensor, ...] | None = None
        self._render_frame_cursor: int = 0
        if self.eye_blink_enabled:
            self._init_eye_blink_composite()

        # Moshi models for Helium extraction
        self._init_moshi(args)

        # Optional local audio file (simulate-live mode)
        self.audio_pcm: np.ndarray | None = None
        if getattr(args, "audio_path", "") and Path(args.audio_path).is_file():
            self.audio_pcm = load_audio_24k(args.audio_path)
            print(
                f"[liveTryHeliumFM] audio_path loaded: {self.audio_pcm.shape[0]/TARGET_SR:.2f}s "
                f"chunk={self.audio_chunk_sec:.3f}s/{self.audio_chunk_samples} samples "
                f"fm_chunk={self.fm_chunk_frames}f",
                flush=True,
            )

        # Shared noise tensor (pre-generated, indexed by absolute frame position)
        self.noise_buf: torch.Tensor | None = None
        if getattr(args, "shared_noise", False):
            max_frames = int(getattr(args, "noise_max_frames", 5000))
            gen = torch.Generator(device=self.device)
            gen.manual_seed(int(getattr(args, "noise_seed", 1234)))
            self.noise_buf = torch.randn(
                1, max_frames, int(args.dim_w), device=self.device, generator=gen
            )
            print(f"[liveTryHeliumFM] shared noise buf: {tuple(self.noise_buf.shape)}", flush=True)

        # Per-session state (reset on each new client)
        self.stream_state: dict | None = None
        self.abs_frame: int = 0
        self.helium_context_tail: torch.Tensor | None = None
        self.helium_deque_size = max(
            1, int(getattr(args, "helium_deque_size", 100))
        )
        self.helium_deque: torch.Tensor | None = None
        self.helium_deque_filled: int = 0
        self.silence_helium_seed: torch.Tensor | None = None
        silence_helium_path = str(
            getattr(args, "silence_helium_path", "") or ""
        )
        if silence_helium_path:
            payload = torch.load(silence_helium_path, map_location="cpu")
            seed = (
                payload.get("silence_helium_mean")
                if isinstance(payload, dict)
                else payload
            )
            if not isinstance(seed, torch.Tensor) or seed.numel() != 4096:
                raise RuntimeError(
                    f"Invalid silence Helium seed: {silence_helium_path}"
                )
            self.silence_helium_seed = seed.reshape(1, 4096).to(
                device=self.device, dtype=torch.float32
            )
            print(
                f"[liveTryHeliumFM] silence Helium seed loaded: "
                f"{silence_helium_path}",
                flush=True,
            )
            runtime_logging.log_event(
                _syslog, "IMTalker.SilenceHeliumSeed", "loaded", path=silence_helium_path,
            )
        self._pcm_accum: np.ndarray = np.empty(0, dtype=np.float32)
        self.dump_motion = bool(getattr(args, "dump_motion", False))
        self.dump_dir = Path(getattr(args, "dump_dir", ROOT / "live_try_dumps"))
        self._session_motion_parts: list[torch.Tensor] = []
        self._session_helium_parts: list[torch.Tensor] = []
        self._session_adapter_50_parts: list[torch.Tensor] = []
        self._session_adapter_25_parts: list[torch.Tensor] = []
        self._session_projected_audio_parts: list[torch.Tensor] = []
        self._session_audio_parts: list[np.ndarray] = []
        self._session_live_token_parts: list[torch.Tensor] = []
        self._session_chunk_rows: list[dict] = []
        self._session_reply_events: list[dict] = []
        self._session_started_wall: float = time.time()

        # JPEG encoding thread pool (CPU-only work, parallelizable)
        self._jpeg_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="jpeg"
        )

        # Warmup
        self._warmup()

        print(
            f"[liveTryHeliumFM] ready — total startup {_ms(t_total):.0f}ms "
            f"fm_chunk={self.fm_chunk_frames} render_sub={self.render_sub_batch} "
            f"dtype={self.dtype}",
            flush=True,
        )
        gpu_fields: dict = {}
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            free_b, total_b = torch.cuda.mem_get_info(idx)
            gpu_fields = {
                "gpu_name": torch.cuda.get_device_name(idx),
                "cuda_version": torch.version.cuda,
                "vram_total_gb": round(total_b / (1024 ** 3), 2),
                "vram_free_gb": round(free_b / (1024 ** 3), 2),
            }
        runtime_logging.log_event(
            _syslog, "IMTalker.LiveHeliumFMEngine", "ready",
            duration_ms=round((time.perf_counter() - _t_engine_init) * 1000.0, 1),
            device=str(self.device), dtype=str(self.dtype),
            fm_chunk_frames=self.fm_chunk_frames, render_sub_batch=self.render_sub_batch,
            avatar_ref_path=str(getattr(args, "ref_path", "")),
            renderer_path=str(getattr(args, "renderer_path", "")),
            generator_path=str(getattr(args, "generator_path", "")),
            **gpu_fields,
        )

    def _init_moshi(self, args: argparse.Namespace) -> None:
        if bool(getattr(args, "direct_reply_hidden", False)) and bool(getattr(args, "enable_moshi_reply", False)):
            self.extractor = None
            print("[liveTryHeliumStudioFM] using direct Moshi reply hidden; batch Helium extractor skipped", flush=True)
            return

        from generate_helium import load_mimi_and_lm

        t0 = time.perf_counter()
        helium_mimi, helium_lm, _ = load_mimi_and_lm(args)
        helium_mimi.eval()
        helium_lm.eval()
        print(
            f"[liveTryHeliumFM] Moshi loaded in {_ms(t0):.0f}ms "
            f"dtype={next(helium_lm.parameters()).dtype}",
            flush=True,
        )

        self.extractor = HeliumExtractor(helium_mimi, helium_lm, self.device)

    def reset_session(self) -> None:
        """Call when a new WebSocket client connects or sends 'start'."""
        self.stream_state = None
        self.abs_frame = 0
        self._render_frame_cursor = 0
        self.helium_context_tail = None
        self.helium_deque = None
        self.helium_deque_filled = 0
        self._pcm_accum = np.empty(0, dtype=np.float32)
        if self.extractor is not None:
            self.extractor.reset()
        self._session_motion_parts = []
        self._session_helium_parts = []
        self._session_adapter_50_parts = []
        self._session_adapter_25_parts = []
        self._session_projected_audio_parts = []
        self._session_audio_parts = []
        self._session_live_token_parts = []
        self._session_chunk_rows = []
        self._session_reply_events = []
        self._session_started_wall = time.time()

    @torch.no_grad()
    def _warmup(self) -> None:
        dummy_pcm = np.zeros(self.audio_chunk_samples, dtype=np.float32)

        t0 = time.perf_counter()
        if self.extractor is None:
            raw_steps = max(1, int(round(self.fm_chunk_frames * 12.5 / float(self.fps))))
            if self.silence_helium_seed is not None:
                dummy_helium = self.silence_helium_seed.expand(
                    raw_steps, -1
                ).contiguous()
            else:
                dummy_helium = torch.zeros(
                    raw_steps, 4096, device=self.device, dtype=torch.float32
                )
            print(f"[liveTryHeliumStudioFM][warmup] raw_helium=skipped direct_hidden raw_steps={raw_steps}", flush=True)
        else:
            dummy_helium = self.extractor.extract_raw_chunk(dummy_pcm)
            _sync_cuda()
            print(f"[liveTryHeliumStudioFM][warmup] raw_helium={_ms(t0):.0f}ms", flush=True)
            self.extractor.reset()

        t0 = time.perf_counter()
        motion, _info = self._sample_motion_from_helium(dummy_helium, self.fm_chunk_frames)
        _sync_cuda()
        print(f"[liveTryHeliumFM][warmup] fm={_ms(t0):.0f}ms motion={tuple(motion.shape)}", flush=True)
        self.stream_state = None
        self.abs_frame = 0
        self._render_frame_cursor = 0
        self.helium_context_tail = None
        self.helium_deque = None
        self.helium_deque_filled = 0

        t0 = time.perf_counter()
        dummy_motion = torch.zeros(self.render_sub_batch, 32, device=self.device, dtype=self.dtype)
        _frames, _timings = self._render_motion(dummy_motion)
        _sync_cuda()
        print(f"[liveTryHeliumFM][warmup] renderer={_ms(t0):.0f}ms", flush=True)
        self._render_frame_cursor = 0

        # Warmup JPEG pool
        t0 = time.perf_counter()
        dummy_np = np.zeros((512, 512, 3), dtype=np.uint8)
        _ = encode_jpeg_b64(dummy_np, self.jpeg_quality)
        print(f"[liveTryHeliumFM][warmup] jpeg={_ms(t0):.0f}ms", flush=True)

        self.stream_state = None

    def feed_pcm(self, pcm_s16le_bytes: bytes) -> Optional[tuple[torch.Tensor, dict, np.ndarray]]:
        pcm = np.frombuffer(pcm_s16le_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        return self.feed_pcm_f32(pcm)

    def feed_pcm_f32(self, pcm_f32: np.ndarray) -> Optional[tuple[torch.Tensor, dict, np.ndarray]]:
        pcm = np.asarray(pcm_f32, dtype=np.float32)
        self._pcm_accum = np.concatenate([self._pcm_accum, pcm])
        if self._pcm_accum.shape[0] < self.audio_chunk_samples:
            return None
        chunk = self._pcm_accum[:self.audio_chunk_samples].copy()
        self._pcm_accum = self._pcm_accum[self.audio_chunk_samples:]
        motion, info = self._process_pcm_chunk(chunk, self.fm_chunk_frames)
        self._record_session_chunk(chunk, motion, info)
        return motion, info, chunk

    @torch.no_grad()
    def _process_pcm_chunk(self, pcm_chunk: np.ndarray, target_frames: int) -> tuple[torch.Tensor, dict]:
        timings: dict = {}
        target_frames = max(1, min(int(target_frames), self.fm_chunk_frames))
        if self.extractor is None:
            raise RuntimeError("direct_reply_hidden mode cannot process raw browser audio directly")

        t0 = time.perf_counter()
        helium = self.extractor.extract_raw_chunk(pcm_chunk)
        timings["helium_ms"] = _ms(t0)
        motion, fm_info = self._sample_motion_from_helium(helium, target_frames)
        timings.update(fm_info)
        return motion, timings

    @torch.no_grad()
    def _sample_motion_from_helium(self, helium: torch.Tensor, target_frames: int) -> tuple[torch.Tensor, dict]:
        timings: dict = {}
        t_adapter = time.perf_counter()
        helium = helium.to(self.device, dtype=torch.float32).contiguous()
        target_frames = int(target_frames)
        current_steps = int(helium.shape[0])
        deque_size = int(getattr(self, "helium_deque_size", 100))
        if self.helium_deque is None:
            if self.silence_helium_seed is not None:
                self.helium_deque = self.silence_helium_seed.expand(
                    deque_size, -1
                ).clone()
            else:
                self.helium_deque = torch.zeros(
                    deque_size,
                    helium.shape[1],
                    device=self.device,
                    dtype=torch.float32,
                )
            self.helium_deque_filled = 0
        if current_steps >= deque_size:
            self.helium_deque = helium[-deque_size:].detach().clone()
            self.helium_deque_filled = deque_size
        else:
            self.helium_deque = torch.cat([self.helium_deque[current_steps:], helium], dim=0).contiguous()
            self.helium_deque_filled = min(deque_size, int(self.helium_deque_filled) + current_steps)

        adapter_window_mode = str(
            getattr(self.args, "adapter_window_mode", "tail")
        )
        if adapter_window_mode == "lookahead":
            # Match training: process the full 8-second window at 50 Hz,
            # emit .96 seconds, and retain .48 seconds as future context.
            future_steps = int(getattr(self.args, "adapter_future_steps", 6))
            target_len_50_full = deque_size * 4
            _baseline, _cnn, feat_50_full = self.studio_adapter.forward_single(
                self.helium_deque, target_len_50_full
            )
            if self.live_sliding_window:
                # Type AC: keep the Type A Helium deque + adapter path, but
                # give IMTalker/FM a small [past + current + future] feature
                # window and emit only current frames. This places lookahead in
                # IMTalker instead of discarding future context inside adapter.
                past_25 = int(self.slide_past_frames)
                future_25 = int(self.slide_future_frames)
                current_25 = int(target_frames)
                past_50 = past_25 * 2
                future_50 = future_25 * 2
                current_50 = current_25 * 2
                full_len_50 = int(feat_50_full.shape[0])
                current_end_50 = full_len_50 - future_50
                current_start_50 = current_end_50 - current_50
                window_start_50 = current_start_50 - past_50
                window_end_50 = current_end_50 + future_50
                if window_start_50 < 0 or current_start_50 < 0 or window_end_50 > full_len_50:
                    raise RuntimeError(
                        "Sliding adapter window is out of range: "
                        f"full={full_len_50} start={window_start_50} "
                        f"current_start={current_start_50} end={window_end_50}"
                    )
                feat_50 = feat_50_full[window_start_50:window_end_50].contiguous()
                feat_25 = F.interpolate(
                    feat_50.T.unsqueeze(0),
                    size=past_25 + current_25 + future_25,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T.contiguous()
            else:
                emitted_frames_50 = max(1, current_steps * 4)
                future_frames_50 = max(0, future_steps * 4)
                segment_end = int(feat_50_full.shape[0]) - future_frames_50
                segment_start = segment_end - emitted_frames_50
                if segment_start < 0:
                    raise RuntimeError(
                        "Look-ahead adapter output is shorter than its emit/future region"
                    )
                feat_50 = feat_50_full[segment_start:segment_end].contiguous()
                feat_25 = F.interpolate(
                    feat_50.T.unsqueeze(0),
                    size=target_frames,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T.contiguous()
        else:
            # Legacy behavior: predict at 25 Hz and emit the newest tail.
            target_len_25_full = deque_size * 2
            _baseline, _cnn, feat_25_full = self.studio_adapter.forward_single(
                self.helium_deque, target_len_25_full
            )
            fresh_frames = max(1, current_steps * 2)
            if int(feat_25_full.shape[0]) < fresh_frames:
                raise RuntimeError(
                    f"Deque adapter output too short: got "
                    f"{feat_25_full.shape[0]}, need {fresh_frames}"
                )
            feat_25 = feat_25_full[-fresh_frames:].contiguous()
            feat_50 = feat_25
        if (not self.live_sliding_window) and int(feat_25.shape[0]) != target_frames:
            feat_25 = F.interpolate(
                feat_25.T.unsqueeze(0),
                size=target_frames,
                mode="linear",
                align_corners=False,
            ).squeeze(0).T.contiguous()
        projected_a = self.fm._project_audio(feat_25.unsqueeze(0).float())
        timings["adapter_ms"] = _ms(t_adapter)
        timings["helium_ms"] = timings["adapter_ms"]
        timings["helium_deque_filled"] = int(self.helium_deque_filled)

        data: dict = {"a_feat": feat_25.unsqueeze(0).float(), "ref_x": self.ref_x}
        if self.noise_buf is not None:
            if self.live_sliding_window:
                noise_start = max(0, self.abs_frame - int(self.slide_past_frames))
                noise_end = noise_start + int(feat_25.shape[0])
            else:
                noise_start = self.abs_frame
                noise_end = self.abs_frame + target_frames
            data["noise_init"] = self.noise_buf[:, noise_start:noise_end]
        t_fm = time.perf_counter()
        if self.live_sliding_window:
            motion = self.fm.sample(
                data,
                a_cfg_scale=float(self.args.a_cfg_scale),
                nfe=int(self.args.nfe),
            )
        else:
            motion, self.stream_state = self.fm.sample(
                data,
                a_cfg_scale=float(self.args.a_cfg_scale),
                nfe=int(self.args.nfe),
                stream_state=self.stream_state,
                return_state=True,
            )
        timings["fm_ms"] = _ms(t_fm)

        motion = motion.squeeze(0).detach()
        if self.live_sliding_window:
            start = int(self.slide_past_frames)
            motion = motion[start:start + target_frames]
        else:
            motion = motion[:target_frames]
        ref_blend = float(getattr(self.args, "motion_ref_blend", 0.0) or 0.0)
        if ref_blend > 0.0:
            ref_motion = self.ref_x.detach().float()
            if ref_motion.ndim == 3:
                ref_motion = ref_motion[0]
            if ref_motion.ndim == 1:
                ref_motion = ref_motion.unsqueeze(0)
            if int(ref_motion.shape[0]) == 1:
                ref_motion = ref_motion.expand(int(motion.shape[0]), -1)
            elif int(ref_motion.shape[0]) != int(motion.shape[0]):
                ref_motion = F.interpolate(
                    ref_motion.T.unsqueeze(0),
                    size=int(motion.shape[0]),
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).T
            ref_motion = ref_motion.to(device=motion.device, dtype=motion.dtype)
            blend = max(0.0, min(1.0, ref_blend))
            motion = motion.mul(1.0 - blend).add(ref_motion, alpha=blend)
        timings["helium_feat"] = helium.detach().cpu()
        timings["adapter_feat_50"] = feat_50.detach().cpu()
        timings["adapter_feat_25"] = feat_25.detach().cpu()
        timings["projected_audio"] = projected_a.squeeze(0).detach().cpu()
        timings["frames"] = int(motion.shape[0])
        timings["abs_start"] = self.abs_frame
        self.abs_frame += timings["frames"]
        return motion, timings

    def _record_session_chunk(self, pcm_chunk: np.ndarray, motion: torch.Tensor, info: dict) -> None:
        self._session_audio_parts.append(np.asarray(pcm_chunk, dtype=np.float32).copy())
        self._session_motion_parts.append(motion.detach().float().cpu().clone())
        helium_feat = info.get("helium_feat")
        if isinstance(helium_feat, torch.Tensor):
            self._session_helium_parts.append(helium_feat.float().cpu().clone())
        adapter_feat_50 = info.get("adapter_feat_50")
        if isinstance(adapter_feat_50, torch.Tensor):
            self._session_adapter_50_parts.append(adapter_feat_50.float().cpu().clone())
        adapter_feat_25 = info.get("adapter_feat_25")
        if isinstance(adapter_feat_25, torch.Tensor):
            self._session_adapter_25_parts.append(adapter_feat_25.float().cpu().clone())
        projected_audio = info.get("projected_audio")
        if isinstance(projected_audio, torch.Tensor):
            self._session_projected_audio_parts.append(projected_audio.float().cpu().clone())
        self._session_chunk_rows.append({
            "chunk": len(self._session_chunk_rows) + 1,
            "abs_start": int(info.get("abs_start", 0)),
            "frames": int(info.get("frames", int(motion.shape[0]))),
            "samples": int(len(pcm_chunk)),
            "helium_ms": float(info.get("helium_ms", 0.0)),
            "fm_ms": float(info.get("fm_ms", 0.0)),
        })

    @torch.no_grad()
    def _extract_wav2vec_raw_50hz(self, audio_24k: np.ndarray) -> torch.Tensor:
        arr = np.asarray(audio_24k, dtype=np.float32)
        if arr.ndim != 1 or arr.size == 0:
            return torch.empty((0, 768), dtype=torch.float32)
        wav = torch.from_numpy(arr).view(1, -1)
        wav16 = torchaudio.functional.resample(wav, TARGET_SR, WAV2VEC_SR).squeeze(0).contiguous().numpy()
        inputs = self.wav2vec_feature_extractor(
            wav16,
            sampling_rate=WAV2VEC_SR,
            return_tensors="pt",
            padding=True,
        )
        kwargs = {
            "input_values": inputs.input_values.to(self.device),
        }
        if getattr(inputs, "attention_mask", None) is not None:
            kwargs["attention_mask"] = inputs.attention_mask.to(self.device)
        frontend = self.wav2vec_model.extract_projected_frontend(**kwargs)
        feat = self.wav2vec_model.encode_from_projected_frontend(
            frontend
        ).last_hidden_state.detach().float().cpu()[0].contiguous()
        return feat

    def dump_last_session(self, *, source: str = "") -> Optional[Path]:
        if not self.dump_motion or not self._session_motion_parts:
            return None
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        session_dir = self.dump_dir / "last_session"
        session_dir.mkdir(parents=True, exist_ok=True)

        motion = torch.cat(self._session_motion_parts, dim=0).contiguous()
        audio = np.concatenate(self._session_audio_parts, axis=0) if self._session_audio_parts else np.empty(0, dtype=np.float32)

        motion_path = session_dir / "full_motion.pt"
        helium_path = session_dir / "full_helium_raw.pt"
        adapter_50_path = session_dir / "full_adapter_w2v_50hz.pt"
        adapter_25_path = session_dir / "full_adapter_w2v_25fps.pt"
        wav2vec_50_path = session_dir / "full_wav2vec_50hz.pt"
        projected_audio_path = session_dir / "full_projected_audio_32.pt"
        audio_path = session_dir / "full_moshi_reply_24k.wav"
        live_tokens_path = session_dir / "live_mimi_tokens.pt"
        reply_events_path = session_dir / "reply_events.jsonl"
        reply_text_path = session_dir / "reply_text.txt"
        meta_path = session_dir / "meta.json"
        helium = None
        adapter_50 = None
        adapter_25 = None
        wav2vec_50 = None
        projected_audio = None
        live_tokens = None
        if self._session_helium_parts:
            helium = torch.cat(self._session_helium_parts, dim=0).contiguous()
        if self._session_adapter_50_parts:
            adapter_50 = torch.cat(self._session_adapter_50_parts, dim=0).contiguous()
        if self._session_adapter_25_parts:
            adapter_25 = torch.cat(self._session_adapter_25_parts, dim=0).contiguous()
        if self._session_projected_audio_parts:
            projected_audio = torch.cat(self._session_projected_audio_parts, dim=0).contiguous()
        if self._session_live_token_parts:
            live_tokens = torch.cat(self._session_live_token_parts, dim=2).contiguous()
        if audio.size > 0:
            wav2vec_50 = self._extract_wav2vec_raw_50hz(audio)

        torch.save({
            "motion": motion,
            "chunks": self._session_chunk_rows,
            "fps": float(self.fps),
            "audio_chunk_sec": float(self.audio_chunk_sec),
            "fm_chunk_frames": int(self.fm_chunk_frames),
            "audio_feat_dim": int(getattr(self.args, "audio_feat_dim", 768)),
            "audio_adapter_dim": int(getattr(self.args, "audio_adapter_dim", 512)),
            "wav2vec_sec": float(self.args.wav2vec_sec),
            "ref_path": str(self.args.ref_path),
            "generator_path": str(self.args.generator_path),
            "renderer_path": str(self.args.renderer_path),
            "source": source,
        }, motion_path)
        if helium is not None:
            torch.save({
                "helium": helium,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "audio_chunk_sec": float(self.audio_chunk_sec),
                "fm_chunk_frames": int(self.fm_chunk_frames),
                "audio_feat_dim": int(getattr(self.args, "audio_feat_dim", 4096)),
                "source": source,
            }, helium_path)
        if adapter_50 is not None:
            torch.save({
                "adapter_feat_50": adapter_50,
                "chunks": self._session_chunk_rows,
                "source": source,
            }, adapter_50_path)
        if adapter_25 is not None:
            torch.save({
                "adapter_feat_25": adapter_25,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "source": source,
            }, adapter_25_path)
        if wav2vec_50 is not None:
            torch.save({
                "wav2vec_50hz": wav2vec_50,
                "chunks": self._session_chunk_rows,
                "sample_rate": int(WAV2VEC_SR),
                "source": source,
            }, wav2vec_50_path)
        if projected_audio is not None:
            torch.save({
                "projected_audio": projected_audio,
                "chunks": self._session_chunk_rows,
                "fps": float(self.fps),
                "source": source,
            }, projected_audio_path)
        if live_tokens is not None:
            torch.save({
                "live_mimi_tokens": live_tokens,
                "chunks": self._session_chunk_rows,
                "source": source,
            }, live_tokens_path)
        if audio.size > 0:
            torchaudio.save(str(audio_path), torch.from_numpy(audio).view(1, -1), TARGET_SR)
        if self._session_reply_events:
            with reply_events_path.open("w", encoding="utf-8") as f:
                for row in self._session_reply_events:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            reply_text = "".join(str(row.get("piece", "")) for row in self._session_reply_events)
            reply_text_path.write_text(reply_text, encoding="utf-8")
        meta_path.write_text(json.dumps({
            "source": source,
            "session_started_wall": float(self._session_started_wall),
            "fps": float(self.fps),
            "audio_chunk_sec": float(self.audio_chunk_sec),
            "fm_chunk_frames": int(self.fm_chunk_frames),
            "motion_frames": int(motion.shape[0]),
            "helium_frames": int(helium.shape[0]) if helium is not None else 0,
            "adapter_50_frames": int(adapter_50.shape[0]) if adapter_50 is not None else 0,
            "adapter_25_frames": int(adapter_25.shape[0]) if adapter_25 is not None else 0,
            "wav2vec_50_frames": int(wav2vec_50.shape[0]) if wav2vec_50 is not None else 0,
            "projected_audio_frames": int(projected_audio.shape[0]) if projected_audio is not None else 0,
            "audio_samples": int(audio.shape[0]),
            "audio_seconds": float(audio.shape[0] / TARGET_SR) if audio.size > 0 else 0.0,
            "reply_text_chars": int(sum(len(str(row.get("piece", ""))) for row in self._session_reply_events)),
            "reply_events": int(len(self._session_reply_events)),
            "chunks": self._session_chunk_rows,
            "ref_path": str(self.args.ref_path),
            "generator_path": str(self.args.generator_path),
            "renderer_path": str(self.args.renderer_path),
        }, indent=2), encoding="utf-8")
        print(f"[liveTryHeliumFM] dumped last session -> {session_dir}", flush=True)
        return session_dir

    def _extract_motion_tensor_from_payload(self, payload, path: str) -> torch.Tensor:
        if isinstance(payload, torch.Tensor):
            motion = payload
        elif isinstance(payload, dict):
            candidates = ["motion", "motion_latents", "latents", "full_motion", "pred_motion", "x"]
            motion = None
            for key in candidates:
                value = payload.get(key)
                if isinstance(value, torch.Tensor):
                    motion = value
                    break
            if motion is None:
                tensor_items = [
                    value
                    for value in payload.values()
                    if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[-1] == int(self.args.dim_w)
                ]
                if tensor_items:
                    motion = tensor_items[0]
            if motion is None:
                raise ValueError(
                    f"No motion tensor found in blink_motion_path={path}. "
                    f"Available keys={list(payload.keys())[:20]}"
                )
        else:
            raise TypeError(f"Unsupported blink motion payload type {type(payload)} from {path}")

        motion = motion.detach().float().cpu()
        if motion.ndim == 3 and motion.shape[0] == 1:
            motion = motion[0]
        if motion.ndim != 2 or motion.shape[-1] != int(self.args.dim_w):
            raise ValueError(
                f"Blink motion must have shape (T,{int(self.args.dim_w)}) or (1,T,{int(self.args.dim_w)}); "
                f"got {tuple(motion.shape)} from {path}"
            )
        return motion.contiguous()

    def _make_eye_mask(self, height: int, width: int) -> torch.Tensor:
        yy = torch.linspace(0.0, 1.0, int(height), device=self.device, dtype=self.dtype).view(1, 1, height, 1)
        xx = torch.linspace(0.0, 1.0, int(width), device=self.device, dtype=self.dtype).view(1, 1, 1, width)
        center_y = float(getattr(self.args, "eye_center_y", 0.405))
        radius_x = max(float(getattr(self.args, "eye_radius_x", 0.145)), 1e-6)
        radius_y = max(float(getattr(self.args, "eye_radius_y", 0.070)), 1e-6)
        feather = max(float(getattr(self.args, "eye_feather", 0.10)), 1e-6)

        mask = torch.zeros(1, 1, height, width, device=self.device, dtype=self.dtype)
        for center_x in (
            float(getattr(self.args, "eye_left_x", 0.36)),
            float(getattr(self.args, "eye_right_x", 0.64)),
        ):
            dist = ((xx - center_x) / radius_x).square() + ((yy - center_y) / radius_y).square()
            ellipse = ((1.0 + feather) - dist).clamp(0.0, feather) / feather
            mask = torch.maximum(mask, ellipse)
        return mask.clamp(0.0, 1.0).contiguous()

    @torch.no_grad()
    def _init_eye_blink_composite(self) -> None:
        blink_path = str(getattr(self.args, "blink_motion_path", "") or "")
        if not blink_path:
            raise ValueError("--enable_eye_blink_composite requires --blink_motion_path")
        if not Path(blink_path).is_file():
            raise FileNotFoundError(f"blink_motion_path does not exist: {blink_path}")

        payload = torch.load(blink_path, map_location="cpu")
        blink_motion = self._extract_motion_tensor_from_payload(payload, blink_path)
        blink_maps_parts: list[list[torch.Tensor]] = []
        chunk = max(1, int(getattr(self.args, "render_sub_batch", 8)))
        for start in range(0, int(blink_motion.shape[0]), chunk):
            sub = blink_motion[start:start + chunk].to(self.device, dtype=self.dtype)
            g_sub = self.g_r.expand(int(sub.shape[0]), -1)
            ta_b = self.renderer.adapt(sub, g_sub)
            maps_b = self.renderer.latent_token_decoder(ta_b)
            if not blink_maps_parts:
                blink_maps_parts = [[] for _ in range(len(maps_b))]
            for idx, map_b in enumerate(maps_b):
                blink_maps_parts[idx].append(map_b.detach())

        self._blink_maps = tuple(torch.cat(parts, dim=0).contiguous() for parts in blink_maps_parts)
        self._eye_masks = tuple(self._make_eye_mask(m.shape[-2], m.shape[-1]) for m in self._blink_maps)
        _sync_cuda()
        shapes = [tuple(m.shape) for m in self._blink_maps]
        print(
            f"[liveTryHeliumFM][blink] cached blink maps from {blink_path}: "
            f"frames={int(blink_motion.shape[0])} shapes={shapes}",
            flush=True,
        )

    def _composite_eye_blink_maps(
        self,
        current_maps: tuple[torch.Tensor, ...] | list[torch.Tensor],
        start_frame: int,
        num_frames: int,
    ) -> tuple[torch.Tensor, ...]:
        if self._blink_maps is None or self._eye_masks is None:
            return tuple(current_maps)
        blink_len = int(self._blink_maps[0].shape[0])
        if blink_len <= 0:
            return tuple(current_maps)
        indices = (torch.arange(int(num_frames), device=self.device) + int(start_frame)) % blink_len
        composited: list[torch.Tensor] = []
        for cur, blink_all, mask in zip(current_maps, self._blink_maps, self._eye_masks):
            blink = blink_all.index_select(0, indices).to(device=cur.device, dtype=cur.dtype)
            mask = mask.to(device=cur.device, dtype=cur.dtype)
            composited.append(blink * mask + cur * (1.0 - mask))
        return tuple(composited)

    @torch.no_grad()
    def _render_motion(self, motion: torch.Tensor) -> tuple[np.ndarray, dict]:
        timings: dict = {}
        t_total = time.perf_counter()
        motion = motion.to(self.device, dtype=self.dtype)
        n = int(motion.shape[0])

        render_start = self._render_frame_cursor
        fused = getattr(self.renderer, '_fused_render', None)
        if self.eye_blink_enabled:
            fused = None

        # torch.compile specializes the renderer to its warmup batch shape.
        # Pad the final short sub-batch so a 50-frame chunk does not compile a
        # second graph for the 2-frame tail during the first live response.
        render_motion = motion
        render_n = n
        if fused is not None and n < self.render_sub_batch:
            pad_n = self.render_sub_batch - n
            render_motion = torch.cat(
                [motion, motion[-1:].expand(pad_n, -1)],
                dim=0,
            ).contiguous()
            render_n = self.render_sub_batch

        g_r_sub = self.g_r.expand(render_n, -1)
        m_r_sub = tuple(m.expand(render_n, -1, -1, -1) for m in self.m_r)
        f_r_sub = [f.expand(render_n, -1, -1, -1) for f in self.f_r]
        if fused is not None:
            frames = fused(render_motion, g_r_sub, m_r_sub, f_r_sub)[:n]
        else:
            ta_c = self.renderer.adapt(motion, g_r_sub)
            m_c = self.renderer.latent_token_decoder(ta_c)
            if self.eye_blink_enabled:
                m_c = self._composite_eye_blink_maps(m_c, render_start, n)
            frames = self.renderer.decode(m_c, m_r_sub, f_r_sub)
        self._render_frame_cursor += n

        frames_np = frames.detach().float().clamp(0, 1).mul(255).to(torch.uint8)
        frames_np = frames_np.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        timings["total_ms"] = _ms(t_total)
        return frames_np, timings

    def render_and_encode_subbatch(
        self,
        motion_sub: torch.Tensor,
        audio_slices: list[np.ndarray],
        abs_start: int,
        text_payload: str,
        avatar_chunk_id: int,
        total_gen_ms: float,
    ) -> list[dict]:
        """Render a sub-batch of frames, JPEG-encode in parallel, return packet dicts."""
        frames_np, _render_info = self._render_motion(motion_sub)

        jpeg_futures = []
        for frame_rgb in frames_np:
            jpeg_futures.append(
                self._jpeg_pool.submit(encode_jpeg_bytes, frame_rgb, self.jpeg_quality)
            )

        packets = []
        gen_ms_i = int(round(float(total_gen_ms)))
        sr_i = int(round(float(TARGET_SR)))
        for j, fut in enumerate(jpeg_futures):
            idx = abs_start + j
            audio_slice = audio_slices[j] if j < len(audio_slices) else np.zeros(
                int(round(TARGET_SR / self.fps)), dtype=np.float32
            )
            jpeg_bytes = fut.result()
            output_audio_codec = str(
                getattr(self.args, "output_audio_codec", "pcm")
            ).lower()
            pcm_b = (
                b""
                if output_audio_codec == "opus"
                else _pcm_f32_to_i16_bytes(audio_slice)
            )
            blob = _wsbin.pack_av_frame(
                idx,
                idx + 1,
                gen_ms_i,
                sr_i,
                jpeg_bytes,
                pcm_b,
                text_payload,
                int(avatar_chunk_id),
            )
            packets.append(
                {
                    "frame_number": idx,
                    "ws_kind": "bytes",
                    "data": blob,
                    "audio_pcm": np.asarray(audio_slice, dtype=np.float32).copy(),
                    "t_ready": time.perf_counter(),
                }
            )
        return packets

    def audio_slice(self, frame_idx: int) -> np.ndarray:
        if self.audio_pcm is None:
            frame_samples = int(round(TARGET_SR / self.fps))
            return np.zeros(frame_samples, dtype=np.float32)
        frame_samples = int(round(TARGET_SR / self.fps))
        start = frame_idx * frame_samples
        chunk = self.audio_pcm[start:start + frame_samples]
        if chunk.shape[0] < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - chunk.shape[0]))
        return chunk


# ---------------------------------------------------------------------------
# WebSocket streaming coroutine (file-driven mode, unchanged)
# ---------------------------------------------------------------------------

async def stream_from_file(ws: WebSocket, engine: LiveHeliumFMEngine) -> None:
    """Simulate live streaming using --audio_path, sending frames back over WS."""
    if engine.audio_pcm is None:
        await ws.send_json({"type": "error", "msg": "No --audio_path given for file-streaming mode"})
        return

    audio = engine.audio_pcm
    lookahead_chunks = max(0, int(getattr(engine.args, "file_chunk_lookahead", 0)))
    total_chunks = int(np.ceil(len(audio) / engine.audio_chunk_samples))
    start_wall = time.perf_counter()
    emitted = 0

    print(
        f"[liveTryHeliumFM] stream_from_file: {total_chunks} chunks "
        f"lookahead={lookahead_chunks}",
        flush=True,
    )

    async def _emit_motion_chunk(
        motion: torch.Tensor,
        fm_info: dict,
        chunk_label: int,
        emitted_so_far: int,
    ) -> int:
        helium_ms = float(fm_info["helium_ms"])
        fm_ms = float(fm_info["fm_ms"])
        n_frames = int(motion.shape[0])
        all_frames_np: list[np.ndarray] = []
        render_ms = 0.0
        for sb_start in range(0, n_frames, engine.render_sub_batch):
            sub = motion[sb_start:sb_start + engine.render_sub_batch].to(
                engine.device, dtype=engine.dtype
            )
            frames_np, render_info = engine._render_motion(sub)
            render_ms += float(render_info["total_ms"])
            all_frames_np.extend(frames_np)

        print(
            f"[liveTryHeliumFM][chunk#{chunk_label}] "
            f"helium={helium_ms:.0f}ms fm={fm_ms:.0f}ms "
            f"render={render_ms:.0f}ms frames={n_frames} "
            f"abs_start={fm_info['abs_start']}",
            flush=True,
        )

        for j, frame_rgb in enumerate(all_frames_np):
            idx = emitted_so_far + j
            chunk_id = idx + 1
            audio_b64 = _pcm_f32_to_i16_b64(engine.audio_slice(idx))
            jpeg_b64 = encode_jpeg_b64(frame_rgb, engine.jpeg_quality)

            await ws.send_json({
                "type": "chunk_audio",
                "chunk_id": chunk_id,
                "sample_rate": TARGET_SR,
                "pcm_s16le_b64": audio_b64,
            })
            await ws.send_json({
                "type": "chunk_frame",
                "chunk_id": chunk_id,
                "frame_idx": 0,
                "jpeg_b64": jpeg_b64,
                "moshi_text": (
                    f"Helium+FM | chunk#{chunk_label} "
                    f"helium={helium_ms:.0f}ms fm={fm_ms:.0f}ms "
                    f"render={render_ms:.0f}ms"
                ),
                "server_fps": round(float(engine.fps), 1),
                "chunks_done": chunk_label,
            })
            target_t = start_wall + (idx + 1) / engine.fps
            await asyncio.sleep(max(0.0, target_t - time.perf_counter()))
        return emitted_so_far + len(all_frames_np)

    if lookahead_chunks <= 0:
        for chunk_idx in range(total_chunks):
            pcm_chunk = audio[
                chunk_idx * engine.audio_chunk_samples:(chunk_idx + 1) * engine.audio_chunk_samples
            ]
            pcm_real = pcm_chunk.copy()
            target_frames = int(round(pcm_chunk.shape[0] * engine.fps / TARGET_SR))
            if pcm_chunk.shape[0] < engine.audio_chunk_samples:
                pcm_chunk = np.pad(pcm_chunk, (0, engine.audio_chunk_samples - pcm_chunk.shape[0]))

            motion, fm_info = engine._process_pcm_chunk(pcm_chunk, target_frames)
            engine._record_session_chunk(pcm_real, motion, fm_info)
            emitted = await _emit_motion_chunk(motion, fm_info, chunk_idx + 1, emitted)
    else:
        pending_real: list[np.ndarray] = []
        prefix_audio = np.empty(0, dtype=np.float32)
        chunk_counter = 0

        def _process_exact_pending(pcm_real: np.ndarray) -> tuple[torch.Tensor, dict]:
            nonlocal prefix_audio
            target_frames = int(round(pcm_real.shape[0] * engine.fps / TARGET_SR))
            t0 = time.perf_counter()
            helium = engine.extractor.extract_exact_chunk_from_prefix(
                prefix_audio,
                engine.abs_frame,
                target_frames,
            )
            _sync_cuda()
            fm_info: dict = {"helium_ms": _ms(t0)}
            motion, sample_info = engine._sample_motion_from_helium(helium, target_frames)
            fm_info.update(sample_info)
            return motion, fm_info

        for chunk_idx in range(total_chunks):
            pcm_real = audio[
                chunk_idx * engine.audio_chunk_samples:(chunk_idx + 1) * engine.audio_chunk_samples
            ].copy()
            prefix_audio = np.concatenate([prefix_audio, pcm_real], axis=0)
            pending_real.append(pcm_real)

            while len(pending_real) > lookahead_chunks:
                chunk_counter += 1
                oldest = pending_real.pop(0)
                motion, fm_info = _process_exact_pending(oldest)
                engine._record_session_chunk(oldest, motion, fm_info)
                emitted = await _emit_motion_chunk(motion, fm_info, chunk_counter, emitted)

        while pending_real:
            chunk_counter += 1
            oldest = pending_real.pop(0)
            motion, fm_info = _process_exact_pending(oldest)
            engine._record_session_chunk(oldest, motion, fm_info)
            emitted = await _emit_motion_chunk(motion, fm_info, chunk_counter, emitted)

    await ws.send_json({"type": "stream_end", "total_frames": emitted})
    engine.dump_last_session(source=str(engine.args.audio_path))
    print(f"[liveTryHeliumFM] stream done: {emitted} frames", flush=True)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

class LiveHeliumFMOptions(BaseOptions):
    def initialize(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = super().initialize(parser)
        parser.set_defaults(wav2vec_sec=0.96)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8998)
        parser.add_argument("--html_path", default=str(ROOT / "static" / "index_v3_binary_fullscreen.html"))
        parser.add_argument("--generator_path", required=True)
        parser.add_argument("--lora_generator_path", default="", help="Optional LoRA generator checkpoint to apply on top of --generator_path")
        parser.add_argument("--lora_rank", type=int, default=64)
        parser.add_argument("--lora_alpha", type=float, default=128.0)
        parser.add_argument("--lora_dropout", type=float, default=0.05)
        parser.add_argument("--no_lora_pose_projection", action="store_true")
        parser.add_argument("--no_lora_audio_projection", action="store_true")
        parser.add_argument("--only_lora_pose_projection", action="store_true")
        parser.add_argument("--renderer_path", required=True)
        parser.add_argument("--adapter_path", required=True, help="Frontend fp32 Helium->Wav2Vec2 projected-frontend adapter checkpoint")
        parser.add_argument(
            "--adapter_type",
            choices=("frontend", "unitalk_last_layer"),
            default="frontend",
            help="Adapter architecture. UniTalk passes only layer 12 to IMTalker.",
        )
        parser.add_argument(
            "--adapter_window_mode",
            choices=("tail", "lookahead"),
            default="tail",
        )
        parser.add_argument(
            "--adapter_future_steps",
            type=int,
            default=6,
            help="12.5 Hz future-context steps retained in lookahead mode.",
        )
        parser.add_argument("--adapter_num_layers", type=int, default=6, help="Transformer layers in the frontend adapter checkpoint")
        parser.add_argument("--adapter_dropout", type=float, default=0.1, help="Dropout value used when constructing the frontend adapter")
        parser.add_argument("--stats_path", default="", help="Unused for frontend adapter mode; accepted for compatibility")
        parser.add_argument(
            "--silence_helium_path",
            default="",
            help="Optional real-silence Helium mean used to initialize the deque.",
        )
        parser.add_argument("--ref_path", required=True)
        parser.add_argument("--audio_path", default="", help="WAV to stream in fixed chunks (simulate-live mode)")
        # Moshi
        parser.add_argument("--moshi_root", default="/workspace/moshi")
        parser.add_argument("--mimi_hf_repo", default="kyutai/moshiko-pytorch-bf16")
        parser.add_argument("--moshi_weight", default="", help="Optional local PersonaPlex/Moshi LM checkpoint")
        parser.add_argument("--mimi_weight", default="", help="Optional local Mimi checkpoint")
        parser.add_argument("--tokenizer", default="", help="Optional local sentencepiece tokenizer")
        parser.add_argument("--quantize_4bit", action="store_true", help="Load PersonaPlex/Moshi LM with bnb 4-bit quantization")
        parser.add_argument("--num_codebooks", type=int, default=8, help="PersonaPlex/Moshi audio codebooks")
        parser.add_argument("--moshi_context", type=int, default=0, help="Optional PersonaPlex/Moshi KV context length")
        parser.add_argument("--voice_prompt", default="", help="PersonaPlex voice prompt filename, e.g. NATM0.pt")
        parser.add_argument("--voice_prompt_dir", default="", help="Optional PersonaPlex voice prompt directory")
        parser.add_argument("--text_prompt", default="", help="Optional PersonaPlex system text prompt")
        parser.add_argument("--moshi_reply_device", default=None, help="Optional separate CUDA device for Moshi reply generation")
        parser.add_argument("--enable_moshi_reply", action="store_true", help="Mic -> Moshi reply audio -> Helium/FM avatar")
        parser.add_argument("--moshi_cfg_coef", type=float, default=1.0)
        parser.add_argument("--direct_reply_hidden", action="store_true", default=True, help="Use Moshi generation hidden directly instead of re-encoding reply audio")
        parser.add_argument("--no_direct_reply_hidden", dest="direct_reply_hidden", action="store_false")
        parser.add_argument(
            "--disable_assistant_output_gate",
            action="store_true",
            help="Disable assistant-output RMS gating and always drive IMTalker from live PersonaPlex hidden states",
        )
        parser.add_argument(
            "--assistant_speech_rms_threshold",
            type=float,
            default=0.006,
            help="RMS threshold on PersonaPlex assistant reply audio before hidden states are allowed to drive avatar motion",
        )
        parser.add_argument(
            "--assistant_speech_hold_chunks",
            type=int,
            default=1,
            help="Keep assistant gate open for this many avatar chunks after reply audio drops below threshold",
        )
        # FM
        parser.add_argument("--audio_chunk_sec", type=float, default=0.96)
        parser.add_argument("--fm_chunk_frames", type=int, default=24, help="Must match wav2vec_sec×fps")
        parser.add_argument(
            "--helium_deque_size",
            type=int,
            default=100,
            help="Number of 12.5 Hz PersonaPlex hidden steps exposed to the adapter.",
        )
        parser.add_argument(
            "--enable_live_sliding_window",
            action="store_true",
            help="Run IMTalker/FM on a past/current/future feature window and emit only the current slice.",
        )
        parser.add_argument(
            "--slide_past_frames",
            type=int,
            default=10,
            help="Past 25Hz frames used by --enable_live_sliding_window.",
        )
        parser.add_argument(
            "--slide_future_frames",
            type=int,
            default=3,
            help="Future 25Hz frames used by --enable_live_sliding_window; 3 frames = 120ms.",
        )
        parser.add_argument("--skip_fm_audio_encoder", action="store_true", help="Skip FMGenerator's raw-audio Wav2Vec encoder; live PersonaPlex passes precomputed adapter features")
        parser.add_argument("--reply_hidden_steps_per_chunk", type=int, default=0, help="Raw Moshi 12.5Hz hidden steps per avatar chunk; 0 derives from fm_chunk_frames/fps")
        parser.add_argument("--prebuffer_chunks", type=int, default=3, help="Avatar chunks queued before sender starts pacing")
        parser.add_argument("--frame_q_backpressure", type=int, default=160)
        parser.add_argument(
            "--max_event_backlog_sec", type=float, default=0.6,
            help="Backpressure cap on persona_event_q, the queue where this pipeline's standing "
                 "latency actually accumulates (logs_4: a rock-steady 9.1-9.6s on every turn; "
                 "logs_6: 10.72s while both media queues were normal). The producer waits when the "
                 "queue is this deep instead of racing ahead of the real-time consumer. Nothing is "
                 "dropped -- earlier versions dropped events or media here and that skewed the "
                 "audio/video timelines apart and deleted answers. 0 disables the cap.",
        )
        parser.add_argument(
            "--media_resync_lag_sec", type=float, default=0.0,
            help="DEFAULT OFF. When >0, an idle media resync drops audio_q+frame_q together and "
                 "re-anchors both senders to one epoch once this much media is queued after ~1.5s "
                 "of silence. Disabled because logs_6 showed any threshold near the pipeline's "
                 "normal buffer (a 2.0s chunk published at once, frame_q allowed to 1.28s) makes it "
                 "fire continuously -- it ran every 5s and delivered 228-248 char answers as 6-42 "
                 "audio packets. Only enable with a value well ABOVE normal video_q_latency_s, and "
                 "re-verify audio_packets_streamed on real hardware.",
        )
        parser.add_argument(
            "--suppress_media_watchdog_sec", type=float, default=3.0,
            help="Forensic fix (logs_3, 2026-09-06): a native-barge-in media suppression that never "
                 "sees its exact expected RMS pattern (silence, then a fresh loud reply) used to stay "
                 "suppressed for the REST OF THE SESSION -- every subsequent turn's audio (search or "
                 "not) was silently dropped before ever reaching the frame/audio queues. This caps how "
                 "long suppression can last before it is force-cleared unconditionally.",
        )
        parser.add_argument(
            "--file_chunk_lookahead",
            type=int,
            default=0,
            help="For --audio_path mode, wait this many future chunks before emitting the oldest chunk",
        )
        parser.add_argument("--render_sub_batch", type=int, default=8)
        parser.add_argument("--jpeg_quality", type=int, default=86)
        parser.add_argument("--reply_audio_gain", type=float, default=1.0, help="Accepted for launch-script compatibility")
        parser.add_argument("--enable_seedvc", action="store_true")
        parser.add_argument("--seedvc_repo", default="/home/ubuntu/project/seed-vc")
        parser.add_argument("--seedvc_reference_audio", default="")
        parser.add_argument("--seedvc_steps", type=int, default=6)
        parser.add_argument("--seedvc_cfg", type=float, default=0.7)
        parser.add_argument("--seedvc_prompt_seconds", type=float, default=5.0)
        parser.add_argument(
            "--motion_ref_blend",
            type=float,
            default=0.0,
            help="Blend generated motion latent toward the reference image latent to reduce head tilt.",
        )
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--buffer_ms", type=int, default=80)
        parser.add_argument(
            "--renderer_precision",
            choices=("fp32", "fp16", "bf16"),
            default="fp32",
        )
        parser.add_argument(
            "--output_audio_codec",
            choices=("pcm", "opus"),
            default="pcm",
            help="Assistant audio transport; Opus is one persistent stream per session.",
        )
        parser.add_argument("--dump_motion", action="store_true", help="Dump last session motion/audio to disk")
        parser.add_argument("--dump_dir", default=str(ROOT / "live_try_dumps"))
        # Shared noise
        parser.add_argument("--shared_noise", action="store_true")
        parser.add_argument("--noise_seed", type=int, default=1234)
        parser.add_argument("--noise_max_frames", type=int, default=5000)
        parser.add_argument(
            "--noise_temporal_corr",
            type=float,
            default=0.0,
            help="AR(1)-style temporal correlation applied to FM initial noise.",
        )
        parser.add_argument(
            "--motion_prior_noise_blend",
            type=float,
            default=0.0,
            help="Small blend from previous/generated motion into FM initial noise.",
        )
        # Precision
        parser.add_argument("--fp32", action="store_true")
        parser.add_argument("--tf32", action="store_true")
        parser.add_argument("--compile_renderer", action="store_true")
        # Eye blink motion-map compositing
        parser.add_argument("--enable_eye_blink_composite", action="store_true")
        parser.add_argument("--blink_motion_path", default="", help="Cached blink motion latent .pt file")
        parser.add_argument("--eye_left_x", type=float, default=0.36)
        parser.add_argument("--eye_right_x", type=float, default=0.64)
        parser.add_argument("--eye_center_y", type=float, default=0.405)
        parser.add_argument("--eye_radius_x", type=float, default=0.145)
        parser.add_argument("--eye_radius_y", type=float, default=0.070)
        parser.add_argument("--eye_feather", type=float, default=0.10)
        # STT + query routing + web search (all optional; omitting every flag
        # below reproduces the plain conversational behavior of this script)
        parser.add_argument("--ref_lora_dir", default="", help="Dir containing lora/ with the <lookup>/<ref> LoRA adapter")
        parser.add_argument("--merge_ref_lora", action="store_true", help="Merge the reference LoRA into base weights instead of keeping it unmerged (QLoRA-style; default is unmerged)")
        parser.add_argument("--max_ref_tokens", type=int, default=250, help="Cap on injected <ref> block length, in tokens")
        parser.add_argument(
            "--router_threshold", type=float, default=0.40,
            help="P(needs live data) at/above which a search fires. Below 0.5 on purpose: "
                 "an unnecessary search costs ~2s of thinking sound; a missed one costs a "
                 "confidently wrong spoken answer.",
        )
        parser.add_argument(
            "--router_rules", type=int, default=1,
            help="1 = run the instant regex pre-pass before the router model (obvious cases cost 0ms). 0 = route every turn through the model.",
        )
        parser.add_argument("--stt_hf_repo", default="", help="HF repo for the STT/VAD submodel, e.g. kyutai/stt-1b-en_fr-candle. Omit to disable routing/search (no transcript, no turn-detection signal).")
        parser.add_argument("--stt_pkg_dir", default="", help="Directory containing an isolated `pip install --no-deps --target <dir> moshi` install of the upstream Kyutai moshi package, used only for the STT submodel")
        parser.add_argument("--vad_threshold", type=float, default=0.5)
        parser.add_argument(
            "--suppress_text_during_search", type=int, default=1,
            help="Hold the model silent for the whole search instead of only muting its audio "
                 "(1=on). Muting alone lets it compose an invented figure behind the filler and "
                 "finish that sentence even after the real <ref> arrives.",
        )
        parser.add_argument("--stt_reject_foreign_script", type=int, default=1, help="Drop transcripts the bundled en/fr STT model could not have produced (non-Latin script) before routing/search")
        parser.add_argument("--stt_max_non_latin_ratio", type=float, default=0.15)
        parser.add_argument("--stt_require_english", type=int, default=1, help="Also drop Latin-script transcripts that read as French/Spanish/etc rather than English")
        parser.add_argument("--compressor_model", default="", help="Shared instruct model for BOTH the query router and web-result compression, e.g. Qwen/Qwen2.5-1.5B-Instruct. Omitting this disables routing and search entirely.")
        parser.add_argument("--compressor_device", default="cuda")
        parser.add_argument("--compressor_4bit", type=int, default=1)
        parser.add_argument("--compressor_max_passages", type=int, default=2)
        parser.add_argument("--web_search_enabled", action="store_true", help="Let the router actually reach the web (needs --web_search_api_key). Without this the router still runs, but a 'needs search' turn falls back to the model's own knowledge.")
        parser.add_argument("--web_search_api_key", default=None)
        parser.add_argument("--web_search_provider", default="tavily", choices=["tavily", "serper", "bing"])
        parser.add_argument("--web_search_max_results", type=int, default=3)
        parser.add_argument("--web_search_timeout", type=float, default=3.0)
        parser.add_argument(
            "--web_search_min_score", type=float, default=0.15,
            help="Relevance floor below which a web result is discarded rather than summarized/injected.",
        )
        parser.add_argument(
            "--conversation_log_dir", default="",
            help="Directory for conversation_<session>.{log,jsonl}, detailed_<session>.log, and "
                 "latency_<session>.{log,jsonl}. Console logging happens regardless of this flag.",
        )
        parser.add_argument("--thinking_sound_path", default="", help="Audio clip looped in place of the model's own audio while a web search is in flight")
        parser.add_argument(
            "--search_max_filler_sec", type=float, default=6.0,
            help="Cap on how long the model is held waiting for search+compression before a generic fallback <ref> is injected.",
        )
        parser.add_argument(
            "--compressor_mode", default="extractive_first",
            choices=["extractive_first", "llm_only", "extractive_only"],
            help="How web-search results become the injected <ref> text. 'extractive_first' (default) "
                 "tries a free, ~0ms best-sentence extraction first and only calls the LLM compressor "
                 "when that is not confident -- this is what removes the 2-5s LLM compression latency "
                 "from the common case. 'llm_only' reproduces the old always-call-the-LLM behavior. "
                 "'extractive_only' never calls the LLM (fastest, lowest VRAM, slightly less polished text).",
        )
        parser.add_argument(
            "--extractive_confidence_threshold", type=float, default=0.55,
            help="Minimum keyword-overlap score for the extractive compressor to be used without "
                 "falling back to the LLM (only relevant with --compressor_mode extractive_first).",
        )
        parser.add_argument(
            "--max_suppress_sec", type=float, default=3.0,
            help="Hard cap on how long the model is held forcibly silent (suppress_text_during_search) "
                 "waiting on a slow search/compress, independent of --search_max_filler_sec. When it "
                 "expires, forced silence is released early but the search keeps running and its <ref> "
                 "still gets injected normally once ready.",
        )
        parser.add_argument(
            "--inject_tokens_per_tick", type=int, default=4,
            help="Max <ref>/fallback tokens force-fed into the live context per 80ms _step() tick. "
                 "Spreads a 20-30 token injection across several ticks instead of blocking the "
                 "real-time GPU thread (mic ingestion + avatar rendering) for up to ~1.3s at once.",
        )
        parser.add_argument(
            "--post_inject_watchdog_sec", type=float, default=4.0,
            help="If no real text/audio is produced within this many seconds of a <ref>/fallback "
                 "injection completing, log it immediately (system_runtime.log + conversation.log) "
                 "and close the turn's latency record out, instead of waiting for the next question's "
                 "VAD to notice minutes later.",
        )
        parser.add_argument(
            "--spell_ref_numbers", action="store_true",
            help="Spell numerals out inside the injected <ref> ('three hundred nine dollars and "
                 "thirty-two cents' instead of '309.32 dollars'). OFF by default on RunPod "
                 "evidence: logs_3 gave the model '$309.32' and it spoke $309.32 correctly, while "
                 "logs_4 gave it the spelled-out form and it spoke '$39.32' -- the longer token "
                 "sequence is re-assembled by the model and loses the magnitude. Currency/percent "
                 "SYMBOLS are always converted to words either way; this only controls the digits.",
        )
        parser.add_argument(
            "--ref_audio_drain_sec", type=float, default=2.0,
            help="Seconds of outgoing audio kept masked (thinking sound, or silence if no clip is "
                 "configured) after a <ref>/fallback injection completes. PersonaPlex's audio "
                 "codebooks trail its text stream, so without this the injected reference text is "
                 "briefly SPOKEN ALOUD before the real answer. 0 disables the mask.",
        )
        return parser


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="IMTalker Helium MotionField Deque FM liveTry")
    assets_dir = ROOT / "static" / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    started_at = time.perf_counter()
    html_path = Path(args.html_path)
    engine: LiveHeliumFMEngine | None = LiveHeliumFMEngine(args)
    moshi_engine: MoshiOnlyEngine | None = None
    seedvc: SeedVCStreamingConverter | None = None
    if bool(getattr(args, "enable_seedvc", False)):
        if not args.seedvc_reference_audio:
            raise ValueError("--seedvc_reference_audio is required with --enable_seedvc")
        seedvc = SeedVCStreamingConverter(
            repo=args.seedvc_repo,
            reference_audio=args.seedvc_reference_audio,
            input_sample_rate=TARGET_SR,
            block_seconds=float(args.audio_chunk_sec),
            diffusion_steps=int(args.seedvc_steps),
            cfg_rate=float(args.seedvc_cfg),
            prompt_seconds=float(args.seedvc_prompt_seconds),
            device=args.device,
        )
        print(
            f"[SeedVC] ready load={seedvc.load_seconds:.1f}s "
            f"reference={args.seedvc_reference_audio} steps={args.seedvc_steps} "
            f"cfg={args.seedvc_cfg}",
            flush=True,
        )

    def get_engine() -> LiveHeliumFMEngine:
        nonlocal engine
        if engine is None:
            engine = LiveHeliumFMEngine(args)
        return engine

    def get_moshi_engine() -> MoshiOnlyEngine:
        nonlocal moshi_engine
        if moshi_engine is None:
            moshi_engine = MoshiOnlyEngineWithHidden(
                moshi_root=args.moshi_root,
                mimi_hf_repo=args.mimi_hf_repo,
                device=getattr(args, "moshi_reply_device", None) or args.device,
                cfg_coef=float(args.moshi_cfg_coef),
                placeholder_jpeg_b64="",
                moshi_weight=getattr(args, "moshi_weight", ""),
                mimi_weight=getattr(args, "mimi_weight", ""),
                tokenizer=getattr(args, "tokenizer", ""),
                quantize_4bit=bool(getattr(args, "quantize_4bit", False)),
                num_codebooks=int(getattr(args, "num_codebooks", 8)),
                context=(int(args.moshi_context) if int(getattr(args, "moshi_context", 0)) > 0 else None),
                voice_prompt=getattr(args, "voice_prompt", ""),
                voice_prompt_dir=getattr(args, "voice_prompt_dir", ""),
                text_prompt=getattr(args, "text_prompt", ""),
                ref_lora_dir=getattr(args, "ref_lora_dir", ""),
                merge_ref_lora=bool(getattr(args, "merge_ref_lora", False)),
                max_ref_tokens=int(getattr(args, "max_ref_tokens", 250)),
                router_threshold=float(getattr(args, "router_threshold", 0.40)),
                router_use_rules=bool(int(getattr(args, "router_rules", 1))),
                stt_hf_repo=getattr(args, "stt_hf_repo", ""),
                stt_pkg_dir=getattr(args, "stt_pkg_dir", ""),
                vad_threshold=float(getattr(args, "vad_threshold", 0.5)),
                suppress_text_during_search=bool(int(getattr(args, "suppress_text_during_search", 1))),
                stt_reject_foreign_script=bool(int(getattr(args, "stt_reject_foreign_script", 1))),
                stt_max_non_latin_ratio=float(getattr(args, "stt_max_non_latin_ratio", 0.15)),
                stt_require_english=bool(int(getattr(args, "stt_require_english", 1))),
                compressor_model=getattr(args, "compressor_model", ""),
                compressor_device=getattr(args, "compressor_device", "cuda"),
                compressor_4bit=bool(int(getattr(args, "compressor_4bit", 1))),
                compressor_max_passages=int(getattr(args, "compressor_max_passages", 2)),
                web_search_enabled=bool(getattr(args, "web_search_enabled", False)),
                web_search_api_key=getattr(args, "web_search_api_key", None),
                web_search_provider=getattr(args, "web_search_provider", "tavily"),
                web_search_max_results=int(getattr(args, "web_search_max_results", 3)),
                web_search_timeout=float(getattr(args, "web_search_timeout", 3.0)),
                web_search_min_score=float(getattr(args, "web_search_min_score", 0.15)),
                conversation_log_dir=getattr(args, "conversation_log_dir", ""),
                thinking_sound_path=getattr(args, "thinking_sound_path", ""),
                search_max_filler_sec=float(getattr(args, "search_max_filler_sec", 6.0)),
                compressor_mode=getattr(args, "compressor_mode", "extractive_first"),
                extractive_confidence_threshold=float(getattr(args, "extractive_confidence_threshold", 0.55)),
                max_suppress_sec=float(getattr(args, "max_suppress_sec", 3.0)),
                inject_tokens_per_tick=int(getattr(args, "inject_tokens_per_tick", 4)),
                post_inject_watchdog_sec=float(getattr(args, "post_inject_watchdog_sec", 4.0)),
                ref_audio_drain_sec=float(getattr(args, "ref_audio_drain_sec", 2.0)),
                spell_ref_numbers=bool(getattr(args, "spell_ref_numbers", False)),
            )
        return moshi_engine

    split_sessions: dict[str, dict] = {}
    # PersonaPlex/Mimi streaming state is process-global. Never let a new
    # browser session reset it while the previous session is still tearing down.
    conversation_lock = asyncio.Lock()

    async def _get_split_media_epoch(session: dict) -> float:
        while not session["prebuffer_ready"].is_set() and not session.get("closed", False):
            await asyncio.sleep(0.01)
        async with session["media_epoch_lock"]:
            if session.get("media_epoch") is None:
                session["media_epoch"] = time.perf_counter() + 0.08
            return float(session["media_epoch"])

    def _media_generation(session: dict) -> int:
        with session["media_generation_lock"]:
            return int(session["media_generation"])

    if bool(getattr(args, "enable_moshi_reply", False)):
        print("[liveTryHeliumFM_ws_binary] eager-loading Moshi/PersonaPlex reply engine", flush=True)
        get_moshi_engine()

    @app.get("/")
    async def index():
        if html_path.is_file():
            return FileResponse(
                html_path,
                headers={
                    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
            )
        return HTMLResponse(f"<h1>Missing HTML</h1><p>Expected: {html_path}</p>", status_code=500)

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True,
            "stage": "live_moshi_reply_helium_fm" if args.enable_moshi_reply else "live_helium_fm",
            "uptime_sec": round(time.perf_counter() - started_at, 3),
            "loaded": engine is not None,
        })

    # Shared A/V clock, written by each sender as it releases a packet and
    # sampled by the health logger. Both values are "media seconds emitted
    # since this generation's epoch", so their DIFFERENCE is the real
    # audio/video offset the user perceives -- the one number that says whether
    # lip-sync is holding or drifting apart over a long answer.
    #
    # It lives at build_app scope, NOT inside a handler, because audio and
    # video are streamed by two SEPARATE websocket endpoints
    # (/ws/conversation and /ws/video). Defined inside either one, the other
    # cannot see it -- which is how logs_7 ended up reporting av_offset_ms
    # climbing to 220s: video_media_s was simply never written.
    av_clock = {"audio_media_s": 0.0, "video_media_s": 0.0, "generation": 0}

    @app.websocket("/ws/video")
    async def video_stream(ws: WebSocket):
        await ws.accept()
        session_id = str(ws.query_params.get("session_id", ""))
        session = split_sessions.get(session_id)
        if session is None:
            await ws.send_json({
                "type": "error",
                "message": "unknown or expired video session",
            })
            await ws.close()
            return

        frame_q: asyncio.Queue = session["frame_q"]
        send_start_wall = await _get_split_media_epoch(session)
        active_generation = _media_generation(session)
        base_frame_idx: int | None = None
        frames_sent = 0
        starvation_events = 0
        starve_start: float | None = None
        # Same anti-burst contract the audio sender uses: one frame period
        # minus a little slack, so normal pacing is never slowed but a
        # backlog can never be flushed back to back.
        native_frame_sec = 1.0 / float(args.fps)
        min_frame_interval_s = max(0.001, native_frame_sec - 0.005)
        next_frame_wall = send_start_wall
        print(
            f"[AJ][VIDEO] connected session={session_id[:8]} "
            f"queued={frame_q.qsize()}",
            flush=True,
        )
        try:
            while True:
                try:
                    packet = frame_q.get_nowait()
                except asyncio.QueueEmpty:
                    if starve_start is None:
                        starve_start = time.perf_counter()
                        starvation_events += 1
                    await asyncio.sleep(0.004)
                    if session.get("closed", False) and frame_q.empty():
                        break
                    continue

                if packet is None:
                    break

                packet_generation = int(packet.get("media_generation", 0))
                current_generation = _media_generation(session)
                if packet_generation != current_generation:
                    continue
                if packet_generation != active_generation or base_frame_idx is None:
                    active_generation = packet_generation
                    base_frame_idx = int(packet["frame_number"])
                    send_start_wall = await _get_split_media_epoch(session)
                    next_frame_wall = send_start_wall
                    await ws.send_json({
                        "type": "video_epoch",
                        "generation": active_generation,
                    })
                    print(f"[MEDIA-EPOCH][VIDEO] generation={active_generation} base_frame={base_frame_idx}", flush=True)

                if starve_start is not None:
                    gap_ms = 1000.0 * (time.perf_counter() - starve_start)
                    if gap_ms > 100:
                        print(
                            f"[AJ][VIDEO] STARVED {gap_ms:.0f}ms "
                            f"(event #{starvation_events}) "
                            f"frame_q={frame_q.qsize()} sent={frames_sent}",
                            flush=True,
                        )
                    starve_start = None

                idx = int(packet["frame_number"])
                # Anti-burst floor, matching _audio_sender's min_send_interval_s.
                # Without it, any moment where target_t has fallen into the past
                # -- a render hiccup, GPU contention, the producer briefly
                # falling behind 25 fps -- makes this loop dump every queued
                # frame back to back as fast as the socket accepts them. The
                # browser sees a burst, then starvation, then another burst,
                # while the audio path (which IS throttled) keeps its own pace:
                # video judder plus lip-sync that slips further off with every
                # stall. Both paths have to degrade the same way or they drift.
                target_t = send_start_wall + (idx - base_frame_idx) / float(args.fps)
                target_t = max(target_t, next_frame_wall)
                sleep_s = target_t - time.perf_counter()
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
                next_frame_wall = max(target_t, time.perf_counter()) + min_frame_interval_s
                # Publish this sender's media clock so the health logger can
                # report a REAL audio/video offset. logs_7 reported
                # av_offset_ms growing to 220s purely because video_media_s was
                # never written -- the instrumentation lived in _reply_sender,
                # which is dead code (see below); this handler is the one that
                # actually streams video.
                av_clock["video_media_s"] = (idx - base_frame_idx) / float(args.fps)
                av_clock["generation"] = active_generation

                if packet.get("ws_kind") == "bytes":
                    await ws.send_bytes(packet["data"])
                else:
                    await ws.send_json(packet["msg"])
                frames_sent += 1

                late_s = time.perf_counter() - target_t
                if late_s > 0.5:
                    print(
                        f"[AJ][VIDEO] frame {idx} is {late_s*1000:.0f}ms late",
                        flush=True,
                    )
                if frames_sent % 50 == 0:
                    elapsed = time.perf_counter() - send_start_wall
                    print(
                        f"[AJ][VIDEO] sent={frames_sent} frame={idx} "
                        f"frame_q={frame_q.qsize()} elapsed={elapsed:.1f}s "
                        f"starve_events={starvation_events}",
                        flush=True,
                    )
        except (WebSocketDisconnect, RuntimeError, Exception) as exc:
            print(f"[AJ][VIDEO] closed session={session_id[:8]} {exc!r}", flush=True)
        finally:
            print(f"[AJ][VIDEO] done session={session_id[:8]} sent={frames_sent}", flush=True)

    @app.websocket("/ws/conversation")
    async def conversation(ws: WebSocket):
        await ws.accept()
        if conversation_lock.locked():
            print("[SESSION-STRICT] waiting for previous teardown", flush=True)
        await conversation_lock.acquire()
        print("[SESSION-STRICT] acquired PersonaPlex session lock", flush=True)
        fm_engine = get_engine()
        fm_engine.reset_session()
        if seedvc is not None:
            seedvc.reset()
        reply_engine = get_moshi_engine() if args.enable_moshi_reply else None
        # PersonaPlex is reset exactly once at the Start boundary below. Doing
        # it here as well replays uncached prompts twice for every connection.
        browser_input_sr = 48000
        opus_reader = sphn.OpusStreamReader(TARGET_SR)
        audio_packets_seen = 0

        # -- Queues for the reply pipeline --
        # mic_q: (raw_bytes, input_sr) or None to stop
        # frame_q: per-frame packet dicts or None to stop
        mic_q: queue.Queue[tuple[bytes, int, int, float] | None] | None = None
        frame_q: asyncio.Queue[dict | None] | None = None
        audio_q: asyncio.Queue[dict | None] | None = None
        gpu_thread: threading.Thread | None = None
        mic_ingest_thread: threading.Thread | None = None
        persona_thread: threading.Thread | None = None
        sender_task: asyncio.Task | None = None
        audio_sender_task: asyncio.Task | None = None
        event_loop: asyncio.AbstractEventLoop | None = None
        ws_send_lock = asyncio.Lock()
        media_epoch_lock = asyncio.Lock()
        media_epoch: float | None = None
        session_id = uuid.uuid4().hex
        prebuffer_ready = threading.Event()
        media_generation_lock = threading.Lock()
        media_generation = 0
        playback_interrupt_event = threading.Event()
        playback_state = {"active": False, "hold": 0}
        mic_overlap_packets = 0
        session_started = threading.Event()
        last_mic_level_log_wall = 0.0

        stream_task: asyncio.Task | None = None

        if reply_engine is not None:
            # Match native PersonaPlex: microphone ingress must never be dropped
            # because video delivery is temporarily slower than generation.
            mic_q = queue.Queue()
            frame_q = asyncio.Queue(maxsize=512)
            audio_q = asyncio.Queue(maxsize=512)
            event_loop = asyncio.get_running_loop()
            reply_input_lock = threading.Lock()
            persona_event_q: queue.Queue[dict] = queue.Queue()
            pipeline_stop_event = threading.Event()
            split_sessions[session_id] = {
                "frame_q": frame_q,
                "prebuffer_ready": prebuffer_ready,
                "media_epoch_lock": media_epoch_lock,
                "media_epoch": media_epoch,
                "media_generation_lock": media_generation_lock,
                "media_generation": media_generation,
                "closed": False,
                "created_at": time.perf_counter(),
            }

        await ws.send_json({
            "type": "server_ready",
            "variant": (
                "AJ-NETWORK-ISO-CACHED-FP32"
                if os.environ.get("IMTALKER_CACHED_ENGINE", "0").strip().lower()
                in {"1", "true", "yes", "on"}
                else "AJ-NETWORK-ISO-FP32"
            ),
            "session_id": session_id,
            "video_ws_path": f"/ws/video?session_id={session_id}",
            "sample_rate": TARGET_SR,
            "output_audio_codec": str(args.output_audio_codec),
            "model_type": "moshi_reply+helium_fm+renderer" if args.enable_moshi_reply else "helium_fm+renderer",
            "tokens_per_chunk": int(args.fm_chunk_frames),
            "has_audio_file": fm_engine.audio_pcm is not None,
            "buffer_ms": int(args.buffer_ms),
            "av_transport": "binary",
            "target_fps": round(float(args.fps), 2),
        })
        print("[liveTryHeliumFM] websocket connected; sent server_ready", flush=True)

        if reply_engine is not None:
            def _mic_ingest_worker() -> None:
                """Drain browser audio independently from video/render progress."""
                assert mic_q is not None
                packets = 0
                while True:
                    item = mic_q.get()
                    if item is None:
                        break
                    raw_bytes, input_sr, packet_seq, received_at = item
                    if not raw_bytes:
                        continue
                    with reply_input_lock:
                        reply_engine.append_browser_pcm(
                            np.frombuffer(raw_bytes, dtype=np.int16), input_sr
                        )
                    packets += 1
                    queue_age_ms = 1000.0 * (time.perf_counter() - received_at)
                    if queue_age_ms > 100.0:
                        print(
                            f"[OPUS-TRANSPORT] seq={packet_seq} "
                            f"ingest_age={queue_age_ms:.1f}ms pending_q={mic_q.qsize()}",
                            flush=True,
                        )
                    if packets % 250 == 0:
                        print(
                            f"[MIC-ISOLATED] ingested={packets} pending_q={mic_q.qsize()}",
                            flush=True,
                        )
                print(f"[MIC-ISOLATED] stopped ingested={packets}", flush=True)

            def _persona_buffered_samples() -> int:
                with reply_input_lock:
                    return int(reply_engine.input_buffer.shape[0])

            def _persona_process_ready(max_steps: int) -> list[dict]:
                with reply_input_lock:
                    return reply_engine.process_ready_steps_limited(max_steps)

            def _persona_priority_worker() -> None:
                """Run one PersonaPlex step whenever one complete Mimi frame is ready.

                Keeping the critical section to one 80 ms step lets microphone
                ingestion append new PCM between steps instead of waiting behind a
                multi-step catch-up burst.
                """
                priority_stream = None
                if torch.cuda.is_available():
                    try:
                        # PyTorch accepts -1 as a high-priority stream even in builds
                        # that do not expose get_stream_priority_range().
                        priority_stream = torch.cuda.Stream(priority=-1)
                        print("[PERSONA-CONTINUOUS] high-priority CUDA stream enabled", flush=True)
                    except Exception as exc:
                        print(f"[PERSONA-CONTINUOUS] stream fallback: {exc!r}", flush=True)
                total_events = 0
                native_audio_seq = 0
                suppress_media = False
                suppression_silence_confirmed = False
                suppression_silence_steps = 0
                # Forensic fix (logs_3, 2026-09-06): a barge-in that never sees
                # its exact expected RMS pattern (3 near-silent steps, then one
                # >=0.012 "fresh reply" step) left suppress_media stuck True for
                # the REST OF THE SESSION -- every event from that point was
                # silently dropped before reaching persona_event_q, so no audio
                # was ever enqueued again (turns 7, 8, 9 in that log: 0 packets
                # each, search and non-search turns alike -- proving it was not
                # a search-specific bug but this suppression state leaking
                # across turns). This wall-clock cap guarantees suppression
                # cannot outlive one turn no matter what RMS pattern follows.
                suppress_media_started_perf: float | None = None
                SUPPRESS_MEDIA_MAX_SEC = float(getattr(args, "suppress_media_watchdog_sec", 3.0))
                consecutive_silent_events = 0
                last_health_log = 0.0
                _HEALTH_LOG_INTERVAL_S = 5.0
                # ~1.5s of unbroken silence before an idle resync is allowed.
                _BACKLOG_SILENCE_RUN = int(round(1.5 * TARGET_SR / MIMI_FRAME_SIZE))
                _BACKLOG_SILENCE_RMS = 0.003
                # Idle media resync -- DEFAULT OFF (logs_6). It was previously
                # wired to max_event_backlog_sec (0.6s), which is far BELOW the
                # pipeline's designed steady-state media buffer: the GPU thread
                # publishes a whole 2.0s chunk at once (50 frames) and frame_q
                # is deliberately allowed to hold up to frame_q_backpressure
                # (32 frames = 1.28s). logs_6 measured video_q_latency_s at
                # 1.2-4.16s in NORMAL operation, so the condition was
                # permanently true and the resync fired every 5s -- the rate
                # limiter, not the symptom. Each fire dropped audio_q and
                # frame_q, so answers of 228-248 chars were delivered as 6-42
                # packets (0.5-3.4s): the reported "every answer is muted",
                # plus constant generation bumps that shredded the video.
                #
                # It also targeted the wrong queue. logs_6 shows the standing
                # backlog lives in persona_event_q (event_q_latency_s up to
                # 10.72s) which a media resync never touches, while audio_q sat
                # at 0.0-0.4s the whole run. That backlog is now handled by
                # real backpressure below -- no dropping, so nothing can go
                # missing and the A/V timeline is never skewed.
                _RESYNC_MAX_LAG_S = max(0.0, float(getattr(args, "media_resync_lag_sec", 0.0)))
                _RESYNC_MIN_INTERVAL_S = 5.0
                last_resync_perf = 0.0
                resync_count = 0
                # Cap on persona_event_q depth, in 80ms events. This is the
                # queue logs_6 caught growing to 10.72s; holding the producer
                # here bounds standing latency without discarding anything.
                _EVENT_Q_MAX = max(
                    0,
                    int(round(
                        float(getattr(args, "max_event_backlog_sec", 0.6))
                        * TARGET_SR / MIMI_FRAME_SIZE
                    )),
                )
                # Hard ceiling on any single hold. Deliberately SHORT: the gate
                # is evaluated from the previous iteration, so an answer can
                # start while a hold is already running, and whatever remains
                # of that hold delays the answer's onset. Throttling an idle
                # backlog does not need a long hold -- a short one simply
                # re-enters next iteration while the model stays idle -- but a
                # long one would cost real time at exactly the wrong moment.
                _EVENT_Q_HOLD_MAX_S = 0.2
                print(
                    f"[EVENT-BACKLOG] backpressure cap: {_EVENT_Q_MAX} events "
                    f"(~{_EVENT_Q_MAX * 0.08:.2f}s), idle-gated, max hold "
                    f"{_EVENT_Q_HOLD_MAX_S:.1f}s; 0 disables",
                    flush=True,
                )
                print(
                    f"[MEDIA-RESYNC] idle resync threshold: {_RESYNC_MAX_LAG_S:.2f}s of queued media "
                    f"after ~{_BACKLOG_SILENCE_RUN * 0.08:.1f}s of silence; 0 disables",
                    flush=True,
                )

                async def _cancel_stale_media(generation: int) -> None:
                    nonlocal media_epoch
                    dropped_audio = 0
                    dropped_frames = 0
                    while True:
                        try: item = audio_q.get_nowait()
                        except asyncio.QueueEmpty: break
                        if item is not None: dropped_audio += 1
                    while True:
                        try: item = frame_q.get_nowait()
                        except asyncio.QueueEmpty: break
                        if item is not None: dropped_frames += 1
                    async with media_epoch_lock:
                        media_epoch = None
                        split_sessions[session_id]["media_epoch"] = None
                    playback_state["active"] = False
                    playback_state["hold"] = 0
                    async with ws_send_lock:
                        await ws.send_json({"type":"media_reset","generation":generation,"reason":"native_personaplex_barge_in"})
                    print(f"[MEDIA-RESET] generation={generation} dropped_audio={dropped_audio} dropped_frames={dropped_frames}", flush=True)

                def _trigger_media_reset() -> int:
                    with media_generation_lock:
                        split_sessions[session_id]["media_generation"] += 1
                        generation = int(split_sessions[session_id]["media_generation"])
                    prebuffer_ready.clear()
                    asyncio.run_coroutine_threadsafe(_cancel_stale_media(generation), event_loop)
                    return generation

                while not pipeline_stop_event.is_set():
                    if not session_started.is_set() or _persona_buffered_samples() < MIMI_FRAME_SIZE:
                        time.sleep(0.002)
                        continue
                    # -- Backpressure, not dropping (logs_6 root cause) --------
                    # The standing latency this pipeline suffers from lives
                    # HERE, in persona_event_q: logs_6 measured it at 134
                    # events == 10.72s while audio_q sat at 0.0-0.4s and
                    # frame_q at its normal 1.2-2.0s. It grows because this
                    # worker is gated only by mic availability, while the GPU
                    # thread downstream is pinned to real time by frame_q
                    # backpressure. At session start mic_q was 166 packets
                    # deep, so this loop raced far ahead of the wall clock and
                    # the gap it opened was never recoverable.
                    #
                    # Two previous attempts fixed it by DISCARDING media (drop
                    # silent events; drop both media queues on a resync). Both
                    # broke streaming, because everything downstream paces on
                    # absolute index-derived schedules -- removing media
                    # removes schedule time without removing wall time, which
                    # skews audio and video apart permanently, and in logs_6
                    # deleted the answers outright.
                    #
                    # Backpressure has neither failure mode: the producer
                    # simply waits for the consumer, so the queue is bounded,
                    # standing latency is capped, NOTHING is ever discarded,
                    # and the A/V timeline is never touched. Mic audio that
                    # arrives meanwhile accumulates in the engine's own input
                    # buffer, which is already measured and handled.
                    # GATED ON IDLE, and time-bounded (logs_7). PersonaPlex is a
                    # full-duplex model that must keep stepping at 12.5Hz; hold
                    # it and the conversation itself breaks, not just its
                    # timing. logs_7 showed exactly that: frame_q saturates at
                    # FRAME_Q_BACKPRESS (32) as a matter of DESIGN -- the GPU
                    # thread publishes 50 frames per chunk, so it is over the
                    # limit the instant a chunk lands and blocks until the
                    # sender drains it. That block propagates into
                    # persona_event_q, which hit this cap (event_q reached 6,
                    # 7, 7, 8 in the health log), and the worker then stopped
                    # stepping the model. Turn 5 produced answer_chars=0,
                    # answer_text_tokens=0, audio_packets_streamed=0 -- the
                    # model never generated at all.
                    #
                    # So: only hold while the model is IDLE (the startup mic
                    # burst, and the dead air between turns, which is where the
                    # backlog is actually built), never once it is speaking,
                    # and never for longer than one chunk period regardless.
                    if (
                        _EVENT_Q_MAX > 0
                        and consecutive_silent_events >= _BACKLOG_SILENCE_RUN
                    ):
                        _hold_deadline = time.perf_counter() + _EVENT_Q_HOLD_MAX_S
                        while (
                            persona_event_q.qsize() >= _EVENT_Q_MAX
                            and not pipeline_stop_event.is_set()
                            and time.perf_counter() < _hold_deadline
                        ):
                            time.sleep(0.004)

                    buffered_before = _persona_buffered_samples()
                    ready_before = buffered_before // MIMI_FRAME_SIZE
                    if priority_stream is None:
                        events = _persona_process_ready(1)
                    else:
                        with torch.cuda.stream(priority_stream):
                            events = _persona_process_ready(1)
                        # Hidden/audio copies produced by _step synchronize the
                        # generated event before it is published to IMTalker.
                        priority_stream.synchronize()
                    for event in events:
                        input_rms = float(event.get("input_rms", 0.0) or 0.0)
                        reply_rms = float(event.get("reply_rms", 0.0) or 0.0)
                        event_force_idle = bool(event.get("force_idle"))
                        if playback_interrupt_event.is_set():
                            playback_interrupt_event.clear()
                            generation = _trigger_media_reset()
                            suppress_media = True
                            suppression_silence_confirmed = False
                            suppression_silence_steps = 0
                            suppress_media_started_perf = time.perf_counter()
                            print(
                                f"[PLAYBACK-BARGE] generation={generation} "
                                f"step={event.get('step',-1)} input_rms={input_rms:.5f} "
                                f"reply_rms={reply_rms:.5f}",
                                flush=True,
                            )

                        if suppress_media:
                            # Watchdog FIRST, unconditionally: no RMS pattern
                            # below gets a chance to leave suppression stuck
                            # past this deadline, regardless of what the
                            # thinking sound / masked audio / a quiet real
                            # answer happens to measure.
                            if (
                                suppress_media_started_perf is not None
                                and (time.perf_counter() - suppress_media_started_perf) > SUPPRESS_MEDIA_MAX_SEC
                            ):
                                print(
                                    f"[PLAYBACK-BARGE] WATCHDOG force-clearing suppression after "
                                    f"{SUPPRESS_MEDIA_MAX_SEC:.1f}s stuck -- resuming media unconditionally "
                                    f"step={event.get('step',-1)}",
                                    flush=True,
                                )
                                suppress_media = False
                                suppression_silence_confirmed = False
                                suppression_silence_steps = 0
                                suppress_media_started_perf = None
                            elif event_force_idle:
                                # Thinking sound / post-injection mask: not the
                                # model's own content either way. Drop it (media
                                # stays suppressed) without letting its RMS
                                # feed the silence/resume state machine --
                                # otherwise the loud thinking-sound clip could
                                # never look "silent enough" to confirm, or
                                # (worse) could look like the "fresh reply"
                                # that ends suppression before the model has
                                # actually said anything.
                                continue
                            elif not suppression_silence_confirmed:
                                if reply_rms <= 0.003:
                                    suppression_silence_steps += 1
                                else:
                                    suppression_silence_steps = 0
                                if suppression_silence_steps >= 3:
                                    suppression_silence_confirmed = True
                                    print(
                                        f"[PLAYBACK-BARGE] native silence confirmed "
                                        f"step={event.get('step',-1)}",
                                        flush=True,
                                    )
                                continue
                            elif reply_rms < 0.012:
                                continue
                            else:
                                suppress_media = False
                                suppression_silence_confirmed = False
                                suppression_silence_steps = 0
                                suppress_media_started_perf = None
                                print(
                                    f"[PLAYBACK-BARGE] fresh reply starts "
                                    f"step={event.get('step',-1)} reply_rms={reply_rms:.5f}",
                                    flush=True,
                                )

                        if suppress_media:
                            continue

                        # -- Idle media resync (replaces the logs_5 event drain) --
                        #
                        # The problem being solved is still the one logs_4
                        # measured: a standing ~9.4s backlog that forms during
                        # startup warmup and can never drain, because the GPU
                        # thread is pinned to exactly real-time by frame_q
                        # backpressure and so consumes events at the same
                        # 12.5/s the mic produces them.
                        #
                        # The PREVIOUS attempt -- dropping individual silent
                        # events here -- fixed the latency and broke streaming,
                        # and the mechanism is worth spelling out because it is
                        # not obvious. BOTH senders pace off an ABSOLUTE,
                        # index-derived schedule anchored at media_epoch:
                        #     audio: epoch + (idx - base) * 0.08
                        #     video: epoch + idx / fps
                        # That is only correct while packet INDEX advances in
                        # lockstep with WALL TIME. Every dropped event removes
                        # 0.08s of schedule time while removing no wall time,
                        # so after N drops both senders are N*0.08s behind
                        # their own epoch -- permanently, since the epoch was
                        # never re-anchored. Once target_t is in the past the
                        # two diverge, because their catch-up behaviour
                        # differs: the audio sender is throttled to one packet
                        # per min_send_interval_s (0.075s) and therefore drains
                        # 80ms of audio per 75ms of wall clock -- 6.7% fast,
                        # forever -- while the video sender had no throttle at
                        # all and dumped frames back to back. Result: audio
                        # skipping, frames freezing, and an A/V offset that
                        # GROWS through every answer and compounds on every
                        # idle gap. Exactly the "continuous buffering" report.
                        #
                        # The correct primitive already exists in this file and
                        # is proven in the barge-in path: _trigger_media_reset()
                        # -> _cancel_stale_media() drops BOTH queues atomically,
                        # sets media_epoch = None so both senders re-anchor to
                        # ONE common new wall-clock origin via _get_media_epoch(),
                        # bumps media_generation so stale in-flight packets are
                        # discarded consistently, and tells the browser to flush
                        # its own jitter buffer. Latency goes away AND audio and
                        # video stay on a single timeline, which dropping
                        # upstream can never achieve.
                        #
                        # Fired only after sustained silence, so it always lands
                        # in the dead air between turns: never mid-answer, and
                        # never during the thinking sound (force_idle resets the
                        # run -- the model is not idle then, it is waiting).
                        if event_force_idle or reply_rms > _BACKLOG_SILENCE_RMS:
                            consecutive_silent_events = 0
                        else:
                            consecutive_silent_events += 1

                        if _RESYNC_MAX_LAG_S > 0.0 and consecutive_silent_events >= _BACKLOG_SILENCE_RUN:
                            queued_audio_s = (audio_q.qsize() if audio_q is not None else 0) * 0.08
                            queued_video_s = (
                                (frame_q.qsize() / float(args.fps)) if frame_q is not None else 0.0
                            )
                            standing_lag_s = max(queued_audio_s, queued_video_s)
                            now_resync = time.perf_counter()
                            if (
                                standing_lag_s > _RESYNC_MAX_LAG_S
                                and (now_resync - last_resync_perf) > _RESYNC_MIN_INTERVAL_S
                            ):
                                last_resync_perf = now_resync
                                resync_count += 1
                                generation = _trigger_media_reset()
                                consecutive_silent_events = 0
                                print(
                                    f"[MEDIA-RESYNC] idle resync #{resync_count} generation={generation} "
                                    f"standing_lag={standing_lag_s:.2f}s "
                                    f"(audio {queued_audio_s:.2f}s / video {queued_video_s:.2f}s) "
                                    f"-- dropped both queues together and re-anchored the shared "
                                    f"media epoch",
                                    flush=True,
                                )
                                try:
                                    runtime_logging.log_event(
                                        runtime_logging.get_system_logger(), "Pipeline", "media_resync",
                                        generation=generation,
                                        standing_lag_s=round(standing_lag_s, 2),
                                        queued_audio_s=round(queued_audio_s, 2),
                                        queued_video_s=round(queued_video_s, 2),
                                        resync_count=resync_count,
                                    )
                                except Exception:
                                    pass

                        with media_generation_lock:
                            event["media_generation"] = int(split_sessions[session_id]["media_generation"])
                        # Keep native audio private until its complete 2-second
                        # motion/video chunk has rendered. The GPU worker then
                        # publishes matching audio and video as one media epoch.
                        persona_event_q.put(event)
                    total_events += len(events)
                    if ready_before > 2:
                        print(
                            f"[PERSONA-CONTINUOUS] backlog={ready_before} "
                            f"drained={len(events)} event_q={persona_event_q.qsize()}",
                            flush=True,
                        )

                    # -- Pipeline-health telemetry ------------------------
                    # Every queue that can silently accumulate, sampled on a
                    # wall-clock interval (not per event) so it stays cheap,
                    # and routed through the async queued logger rather than
                    # the GPU thread doing any I/O of its own. This is what
                    # makes "it degrades after a few turns" answerable from
                    # the log alone instead of from a live repro.
                    now_health = time.perf_counter()
                    if now_health - last_health_log >= _HEALTH_LOG_INTERVAL_S:
                        last_health_log = now_health
                        try:
                            vram_used_gb = vram_total_gb = None
                            if torch.cuda.is_available():
                                free_b, total_b = torch.cuda.mem_get_info()
                                vram_total_gb = round(total_b / 1e9, 2)
                                vram_used_gb = round((total_b - free_b) / 1e9, 2)
                            # The headline number: both senders report how many
                            # MEDIA seconds they have released since the shared
                            # epoch, so the difference is the actual A/V offset
                            # the viewer perceives. It should hover near zero
                            # and, critically, must not GROW during a long
                            # answer -- that growth is what lip-sync drift is.
                            av_offset_ms = round(
                                (av_clock["audio_media_s"] - av_clock["video_media_s"]) * 1000.0, 1
                            )
                            runtime_logging.log_event(
                                runtime_logging.get_system_logger(), "Pipeline", "health",
                                event_q=persona_event_q.qsize(),
                                frame_q=frame_q.qsize() if frame_q is not None else -1,
                                audio_q=audio_q.qsize() if audio_q is not None else -1,
                                mic_q=mic_q.qsize() if mic_q is not None else -1,
                                event_q_latency_s=round(persona_event_q.qsize() * 0.08, 2),
                                audio_q_latency_s=round((audio_q.qsize() if audio_q is not None else 0) * 0.08, 2),
                                video_q_latency_s=round(
                                    (frame_q.qsize() if frame_q is not None else 0) / float(args.fps), 2
                                ),
                                av_offset_ms=av_offset_ms,
                                audio_media_s=round(av_clock["audio_media_s"], 2),
                                video_media_s=round(av_clock["video_media_s"], 2),
                                media_generation=av_clock["generation"],
                                resync_count=resync_count,
                                suppressed=bool(suppress_media),
                                vram_used_gb=vram_used_gb,
                                vram_total_gb=vram_total_gb,
                            )
                        except Exception:
                            pass
                print(f"[PERSONA-CONTINUOUS] stopped events={total_events}", flush=True)

            def _gpu_producer_thread() -> None:
                """GPU thread: Moshi -> Helium -> FM -> render -> JPEG -> frame_q.

                Runs Moshi at maximum GPU speed. When real mic audio is not
                yet available, pads Moshi input with silence so it can keep
                generating reply audio without waiting for real-time mic
                arrival. This dramatically cuts first-reply latency.
                """
                assert (
                    mic_q is not None
                    and frame_q is not None
                    and audio_q is not None
                    and event_loop is not None
                )
                pending_reply_steps: list[dict] = []
                pending_reply_hidden: list[torch.Tensor] = []
                pending_reply_audio: list[np.ndarray] = []
                reply_step_history: list[dict] = []
                reply_audio_history: list[np.ndarray] = []
                reply_avatar_chunk_idx = 0
                chunk_produce_count = 0
                active_generation = 0
                staged_audio_seq = 0
                # Pause producer when queue is this deep (qsize() is heuristic across threads).
                FRAME_Q_BACKPRESS = max(1, int(getattr(args, "frame_q_backpressure", 96)))
                FRAME_Q_PUT_TIMEOUT_S = 120.0
                prebuffer_chunks = max(0, int(getattr(args, "prebuffer_chunks", PREBUFFER_CHUNKS)))
                hidden_steps_per_chunk = int(getattr(args, "reply_hidden_steps_per_chunk", 0))
                if hidden_steps_per_chunk <= 0:
                    hidden_steps_per_chunk = int(round(float(args.fm_chunk_frames) * 12.5 / float(args.fps)))
                hidden_steps_per_chunk = max(1, hidden_steps_per_chunk)
                # Native PersonaPlex drains every complete Mimi frame currently
                # buffered. A generous cap prevents an unbounded historical
                # backlog from monopolizing the worker after a client stall.
                max_moshi_steps_per_loop = 64
                was_silent = True
                assistant_gate_hold = 0
                last_motion_frame: torch.Tensor | None = None
                seedvc_was_active = False

                def _enqueue_frame(pkt: dict) -> None:
                    """Block until frame_q accepts pkt (real backpressure). Must run from GPU thread."""
                    pkt["media_generation"] = active_generation
                    fut = asyncio.run_coroutine_threadsafe(frame_q.put(pkt), event_loop)
                    try:
                        fut.result(timeout=FRAME_Q_PUT_TIMEOUT_S)
                    except TimeoutError:
                        print(
                            f"[GPU] WARNING frame_q.put timeout frame={pkt.get('frame_number')}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[GPU] WARNING frame_q.put failed: {e!r}", flush=True)

                def _enqueue_audio(pkt: dict) -> None:
                    pkt["media_generation"] = active_generation
                    fut = asyncio.run_coroutine_threadsafe(audio_q.put(pkt), event_loop)
                    try:
                        fut.result(timeout=FRAME_Q_PUT_TIMEOUT_S)
                    except TimeoutError:
                        print(
                            f"[GPU] WARNING audio_q.put timeout frame={pkt.get('frame_number')}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[GPU] WARNING audio_q.put failed: {e!r}", flush=True)

                if prebuffer_chunks <= 0 and not prebuffer_ready.is_set():
                    prebuffer_ready.set()
                    print("[GPU] prebuffer=0, sender starts immediately", flush=True)

                while not pipeline_stop_event.is_set():
                    # Mic ingestion runs in its own worker and remains independent
                    # from all video backpressure below.
                    while frame_q.qsize() >= FRAME_Q_BACKPRESS:
                        time.sleep(0.004)

                    if not session_started.is_set():
                        time.sleep(0.003)
                        continue

                    try:
                        first_event = persona_event_q.get(timeout=0.003)
                    except queue.Empty:
                        time.sleep(0.003)
                        continue

                    # PersonaPlex runs independently; this worker only batches
                    # completed native events for adapter/FM/rendering.
                    t_recv = time.perf_counter()
                    events = [first_event]
                    while len(events) < max_moshi_steps_per_loop:
                        try:
                            events.append(persona_event_q.get_nowait())
                        except queue.Empty:
                            break
                    for ev in events:
                        event_generation = int(ev.get("media_generation", 0))
                        if event_generation != active_generation:
                            active_generation = event_generation
                            pending_reply_steps.clear(); pending_reply_hidden.clear(); pending_reply_audio.clear()
                            reply_step_history.clear(); reply_audio_history.clear()
                            chunk_produce_count = 0; was_silent = True; assistant_gate_hold = 0
                            fm_engine.helium_deque = None
                            fm_engine.helium_deque_filled = 0
                            print(
                                f"[MEDIA-EPOCH][GPU] generation={active_generation} "
                                "cleared adapter batching and Helium window",
                                flush=True,
                            )
                        pending_reply_steps.append(ev)
                        fm_engine._session_reply_events.append({
                            "step": int(ev.get("step", -1)),
                            "token": int(ev.get("token", -1)),
                            "piece": str(ev.get("piece", "")),
                            "audio_text": str(ev.get("audio_text", "")),
                            "reply_rms": float(ev.get("reply_rms", 0.0)),
                            "reply_peak": float(ev.get("reply_peak", 0.0)),
                            "input_rms": float(ev.get("input_rms", 0.0)),
                            "hidden": bool(isinstance(ev.get("helium_hidden"), torch.Tensor)),
                            "total_ms": float(ev.get("total_ms", 0.0)),
                        })
                        reply_pcm = (
                            np.frombuffer(base64.b64decode(ev["reply_i16_b64"]), dtype=np.int16)
                            .astype(np.float32) / 32768.0
                        )
                        if bool(getattr(args, "direct_reply_hidden", False)):
                            hidden = ev.get("helium_hidden")
                            if not isinstance(hidden, torch.Tensor):
                                continue
                            pending_reply_hidden.append(hidden.squeeze(0).contiguous())
                            pending_reply_audio.append(reply_pcm)
                            if len(pending_reply_hidden) < hidden_steps_per_chunk:
                                continue

                            used_hidden = pending_reply_hidden[:hidden_steps_per_chunk]
                            used_audio = pending_reply_audio[:hidden_steps_per_chunk]
                            used_steps = pending_reply_steps[:hidden_steps_per_chunk]
                            pending_reply_hidden = pending_reply_hidden[hidden_steps_per_chunk:]
                            pending_reply_audio = pending_reply_audio[hidden_steps_per_chunk:]
                            pending_reply_steps = pending_reply_steps[hidden_steps_per_chunk:]

                            if str(getattr(args, "adapter_window_mode", "tail")) == "lookahead":
                                history_size = int(fm_engine.helium_deque_size)
                                future_steps = int(
                                    getattr(args, "adapter_future_steps", 6)
                                )
                                reply_audio_history.extend(used_audio)
                                reply_step_history.extend(used_steps)
                                reply_audio_history = reply_audio_history[-history_size:]
                                reply_step_history = reply_step_history[-history_size:]

                                zero_audio = np.zeros_like(used_audio[0])
                                audio_window = (
                                    [zero_audio]
                                    * (history_size - len(reply_audio_history))
                                    + reply_audio_history
                                )
                                silent_step = {
                                    "token": 0,
                                    "reply_rms": 0.0,
                                    "total_ms": 0.0,
                                    "audio_text": "",
                                }
                                step_window = (
                                    [silent_step]
                                    * (history_size - len(reply_step_history))
                                    + reply_step_history
                                )
                                output_end = history_size - future_steps
                                output_start = (
                                    output_end - hidden_steps_per_chunk
                                )
                                used_audio = audio_window[output_start:output_end]
                                used_steps = step_window[output_start:output_end]

                            pcm_chunk = np.concatenate(used_audio, axis=0).astype(np.float32, copy=False)
                            reply_rms = float(np.sqrt(np.mean(np.square(pcm_chunk)))) if pcm_chunk.size else 0.0
                            step_rms = max(
                                (
                                    float(s.get("reply_rms", 0.0) or 0.0)
                                    for s in used_steps
                                ),
                                default=0.0,
                            )
                            speech_threshold = float(
                                getattr(args, "assistant_speech_rms_threshold", 0.006)
                            )
                            # While a web search is in flight, _step() swaps the
                            # model's own audio for a "thinking sound" clip and
                            # marks the event force_idle=True -- that clip's RMS
                            # is well above speech_threshold, so without this
                            # override the avatar would visibly lip-sync to it.
                            any_force_idle = any(bool(s.get("force_idle")) for s in used_steps)
                            assistant_active_now = (
                                max(reply_rms, step_rms) > speech_threshold and not any_force_idle
                            )
                            if any_force_idle:
                                assistant_gate_hold = 0
                            if assistant_active_now:
                                assistant_gate_hold = max(
                                    0,
                                    int(getattr(args, "assistant_speech_hold_chunks", 1)),
                                )
                            elif assistant_gate_hold > 0:
                                assistant_gate_hold -= 1
                            assistant_active = assistant_active_now or assistant_gate_hold > 0
                            if bool(getattr(args, "disable_assistant_output_gate", False)):
                                assistant_active = True

                            previous_active = not was_silent
                            transition = assistant_active != previous_active
                            if transition:
                                print(
                                    f"[GPU] Avatar gate transition "
                                    f"{'speech' if assistant_active else 'idle'} "
                                    f"reply_rms={reply_rms:.5f} step_rms={step_rms:.5f}",
                                    flush=True,
                                )
                            was_silent = not assistant_active

                            if assistant_active:
                                helium_chunk = torch.cat(used_hidden, dim=0)
                            else:
                                if fm_engine.silence_helium_seed is not None:
                                    helium_chunk = fm_engine.silence_helium_seed.expand(
                                        hidden_steps_per_chunk, -1
                                    ).contiguous()
                                else:
                                    hidden_dim = int(used_hidden[0].shape[-1])
                                    helium_chunk = torch.zeros(
                                        hidden_steps_per_chunk,
                                        hidden_dim,
                                        device=fm_engine.device,
                                        dtype=torch.float32,
                                    )
                            target_frames = max(1, int(round(len(pcm_chunk) * float(args.fps) / TARGET_SR)))
                            motion, fm_info = fm_engine._sample_motion_from_helium(helium_chunk, target_frames)
                            if (
                                transition
                                and last_motion_frame is not None
                                and TRANSITION_BLEND_FRAMES > 0
                            ):
                                blend_frames = min(
                                    TRANSITION_BLEND_FRAMES,
                                    int(motion.shape[0]),
                                )
                                alpha = torch.linspace(
                                    0.0,
                                    1.0,
                                    blend_frames + 2,
                                    device=motion.device,
                                    dtype=motion.dtype,
                                )[1:-1].unsqueeze(-1)
                                previous = last_motion_frame.to(
                                    device=motion.device,
                                    dtype=motion.dtype,
                                ).unsqueeze(0)
                                motion[:blend_frames] = (
                                    previous * (1.0 - alpha)
                                    + motion[:blend_frames] * alpha
                                )
                            last_motion_frame = motion[-1].detach().clone()
                            fm_info["assistant_active"] = bool(assistant_active)
                            fm_info["assistant_reply_rms"] = float(reply_rms)
                            fm_info["assistant_step_rms"] = float(step_rms)
                            used_codes = [
                                s["reply_codes"].to(dtype=torch.int16).contiguous()
                                for s in used_steps
                                if isinstance(s.get("reply_codes"), torch.Tensor)
                            ]
                            if used_codes:
                                fm_engine._session_live_token_parts.extend(used_codes)
                            fm_engine._record_session_chunk(pcm_chunk, motion, fm_info)
                        else:
                            result = fm_engine.feed_pcm_f32(reply_pcm)
                            if result is None:
                                continue

                            motion, fm_info, pcm_chunk = result
                            steps_per_avatar_chunk = max(1, int(round(len(pcm_chunk) / MIMI_FRAME_SIZE)))
                            used_steps = pending_reply_steps[:steps_per_avatar_chunk]
                            pending_reply_steps = pending_reply_steps[steps_per_avatar_chunk:]
                            used_codes = [
                                s["reply_codes"].to(dtype=torch.int16).contiguous()
                                for s in used_steps
                                if isinstance(s.get("reply_codes"), torch.Tensor)
                            ]
                            if used_codes:
                                fm_engine._session_live_token_parts.extend(used_codes)

                        reply_avatar_chunk_idx += 1
                        chunk_produce_count += 1

                        moshi_total_ms = sum(float(s.get("total_ms", 0.0)) for s in used_steps)

                        avatar_chunk_id = len(fm_engine._session_chunk_rows)
                        if seedvc is not None:
                            seedvc_active = bool(fm_info.get("assistant_active", True))
                            if seedvc_active:
                                if not seedvc_was_active:
                                    try:
                                        seedvc.reset()
                                    except Exception as exc:
                                        print(f"[SeedVC] reset failed; continuing: {exc!r}", flush=True)
                                try:
                                    pcm_chunk, seedvc_info = seedvc.convert(pcm_chunk)
                                    fm_info.update(seedvc_info)
                                    print(
                                        f"[SeedVC] chunk={reply_avatar_chunk_idx} "
                                        f"ms={seedvc_info['seedvc_ms']:.1f} "
                                        f"rtf={seedvc_info['seedvc_rtf']:.3f} "
                                        f"sola={int(seedvc_info['sola_offset'])}",
                                        flush=True,
                                    )
                                except Exception as exc:
                                    print(f"[SeedVC] conversion failed, using raw PCM: {exc!r}", flush=True)
                            else:
                                pcm_chunk = np.zeros_like(pcm_chunk)
                            seedvc_was_active = seedvc_active
                        frame_audio = split_audio_into_frame_slices(pcm_chunk, args.fps)
                        n_frames = int(motion.shape[0])
                        emitted = int(fm_info["abs_start"])
                        output_step = used_steps[-1] if used_steps else ev
                        text_payload = (
                            output_step.get("audio_text")
                            or output_step.get("sampled_text")
                            or ""
                        )
                        total_gen_ms = (
                            moshi_total_ms + float(fm_info["helium_ms"]) + float(fm_info["fm_ms"])
                        )


                        t_chunk_start = time.perf_counter()
                        staged_frames: list[dict] = []

                        for sb_start in range(0, n_frames, fm_engine.render_sub_batch):
                            sb_end = min(sb_start + fm_engine.render_sub_batch, n_frames)
                            sub_motion = motion[sb_start:sb_end]
                            sub_audio = frame_audio[sb_start:sb_end]

                            packets = fm_engine.render_and_encode_subbatch(
                                sub_motion,
                                sub_audio,
                                abs_start=emitted + sb_start,
                                text_payload=text_payload,
                                avatar_chunk_id=avatar_chunk_id,
                                total_gen_ms=total_gen_ms,
                            )

                            for pkt in packets:
                                staged_frames.append(pkt)

                        # Atomic publication: do not let speech escape before
                        # all matching frames are ready. This intentionally adds
                        # one 2-second generation buffer to preserve A/V sync.
                        # Frames go first so the browser has the complete visual
                        # reserve before audio advances its playback clock.
                        for pkt in staged_frames:
                            _enqueue_frame(pkt)
                        # force_idle marks a step whose outgoing audio is the
                        # thinking sound or the post-injection mask, not the
                        # assistant's answer. It travels with the packet so the
                        # sender does not mistake the waiting cue for the
                        # answer starting to stream (audio_stream_started).
                        # Indexed defensively: used_audio/used_steps are sliced
                        # together on every path that sets both, but one branch
                        # reassigns used_steps alone.
                        masked_flags = [bool(s.get("force_idle")) for s in used_steps]
                        for audio_idx, audio_step in enumerate(used_audio):
                            _enqueue_audio({
                                "audio_packet_index": staged_audio_seq,
                                "audio_pcm": np.asarray(audio_step, dtype=np.float32).copy(),
                                "created_at": time.perf_counter(),
                                "masked": (
                                    masked_flags[audio_idx]
                                    if audio_idx < len(masked_flags) else False
                                ),
                            })
                            staged_audio_seq += 1

                        if not prebuffer_ready.is_set():
                            prebuffer_ready.set()
                            print(
                                f"[GPU][ATOMIC-2S] released audio={len(used_audio)} "
                                f"frames={len(staged_frames)} generation={active_generation}",
                                flush=True,
                            )

                        chunk_wall_ms = _ms(t_chunk_start)
                        produce_latency_ms = _ms(t_recv)
                        q_depth = frame_q.qsize() if frame_q is not None else -1

                        print(
                            f"[GPU][chunk#{reply_avatar_chunk_idx}] "
                            f"moshi={moshi_total_ms:.0f}ms "
                            f"helium={float(fm_info['helium_ms']):.0f}ms "
                            f"fm={float(fm_info['fm_ms']):.0f}ms "
                            f"render+jpeg={chunk_wall_ms:.0f}ms "
                            f"frames={n_frames} "
                            f"produce_latency={produce_latency_ms:.0f}ms "
                            f"frame_q={q_depth} "
                            f"abs={emitted}",
                            flush=True,
                        )

                        # The avatar chunk pipeline is where logs_2 says the
                        # user-perceived delay actually lives: STT was 0.4ms,
                        # routing 0.0ms and the model's first audible frame
                        # landed +0.05s after the question -- yet the answer
                        # was felt seconds later. Everything between those two
                        # points is this block, and until now it existed only
                        # as a stdout print, invisible to the latency report.
                        if reply_engine is not None:
                            turn_id = getattr(reply_engine, "_latency_turn_id", None)
                            if turn_id is not None:
                                lat = reply_engine.conv_logger.latency
                                lat.accumulate(turn_id, "avatar_fm", float(fm_info["fm_ms"]) / 1000.0)
                                lat.accumulate(turn_id, "avatar_helium", float(fm_info["helium_ms"]) / 1000.0)
                                lat.accumulate(turn_id, "avatar_render_jpeg", chunk_wall_ms / 1000.0)
                                if reply_avatar_chunk_idx <= 1:
                                    lat.stage(
                                        turn_id, "avatar_first_chunk_publish",
                                        produce_latency_ms / 1000.0,
                                        note=(
                                            f"audio_chunk_sec={float(getattr(args, 'audio_chunk_sec', 0.0)):.1f} "
                                            f"-- time from the model producing this chunk's audio to the "
                                            f"chunk (audio+{n_frames} frames) being queued for sending"
                                        ),
                                    )

                fut_done = asyncio.run_coroutine_threadsafe(frame_q.put(None), event_loop)
                try:
                    fut_done.result(timeout=30.0)
                except Exception as e:
                    print(f"[GPU] WARNING sentinel put: {e!r}", flush=True)
                fut_audio_done = asyncio.run_coroutine_threadsafe(
                    audio_q.put(None),
                    event_loop,
                )
                try:
                    fut_audio_done.result(timeout=30.0)
                except Exception as e:
                    print(f"[GPU] WARNING audio sentinel put: {e!r}", flush=True)

            async def _get_media_epoch() -> float:
                nonlocal media_epoch
                while not prebuffer_ready.is_set():
                    await asyncio.sleep(0.01)
                async with media_epoch_lock:
                    session = split_sessions.get(session_id)
                    if media_epoch is None and session is not None and session.get("media_epoch") is not None:
                        media_epoch = float(session["media_epoch"])
                    if media_epoch is None:
                        media_epoch = time.perf_counter() + 0.08
                        if session is not None:
                            session["media_epoch"] = media_epoch
                    return media_epoch

            # NOTE: a second frame sender (_reply_sender) used to live here. It was
            # never started -- no asyncio.create_task() ever referenced it -- so it
            # was dead code that silently absorbed two rounds of A/V fixes (logs_6,
            # logs_7 both showed video_media_s stuck at 0.0 because the
            # instrumentation was in here). Video is streamed by the
            # @app.websocket("/ws/video") handler; fix that one.

            async def _audio_sender() -> None:
                assert audio_q is not None
                output_audio_codec = str(args.output_audio_codec).lower()
                if output_audio_codec != "opus":
                    raise RuntimeError("AJ requires --output_audio_codec opus")
                opus_writer = sphn.OpusStreamWriter(TARGET_SR)
                send_start_wall = await _get_media_epoch()
                packets_sent = 0
                bytes_sent = 0
                max_queue_depth = 0
                max_lateness_ms = 0.0
                last_send_wall: float | None = None
                # AG: prevent catch-up bursts after GPU/render stalls.
                # Even if target_t is already late, keep websocket audio writes
                # spaced near the media frame cadence instead of flushing a
                # backlog back-to-back into the browser decoder/worklet.
                # PersonaPlex emits one native Mimi audio step every 80 ms.
                native_packet_sec = float(MIMI_FRAME_SIZE) / float(TARGET_SR)
                min_send_interval_s = max(0.001, native_packet_sec - 0.005)
                next_send_wall = send_start_wall
                active_generation = 0
                base_audio_idx: int | None = None

                while True:
                    packet = await audio_q.get()
                    if packet is None:
                        break
                    session = split_sessions.get(session_id)
                    if session is None: break
                    packet_generation = int(packet.get("media_generation", 0))
                    current_generation = _media_generation(session)
                    if packet_generation != current_generation: continue
                    idx = int(packet["audio_packet_index"])
                    if packet_generation != active_generation or base_audio_idx is None:
                        active_generation = packet_generation
                        base_audio_idx = idx
                        send_start_wall = await _get_media_epoch()
                        next_send_wall = send_start_wall
                        last_send_wall = None
                        opus_writer = sphn.OpusStreamWriter(TARGET_SR)
                        print(f"[MEDIA-EPOCH][AUDIO] generation={active_generation} base_audio={base_audio_idx}", flush=True)
                    target_t = send_start_wall + (idx - base_audio_idx) * native_packet_sec
                    target_t = max(target_t, next_send_wall)
                    sleep_s = target_t - time.perf_counter()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)
                    session = split_sessions.get(session_id)
                    if session is None or packet_generation != _media_generation(session):
                        continue
                    send_wall = time.perf_counter()
                    lateness_ms = max(0.0, (send_wall - target_t) * 1000.0)
                    max_lateness_ms = max(max_lateness_ms, lateness_ms)
                    interval_ms = (
                        0.0 if last_send_wall is None
                        else (send_wall - last_send_wall) * 1000.0
                    )
                    last_send_wall = send_wall
                    next_send_wall = max(target_t, send_wall) + min_send_interval_s
                    max_queue_depth = max(max_queue_depth, audio_q.qsize())
                    av_clock["audio_media_s"] = (idx - base_audio_idx) * native_packet_sec

                    audio_pcm = np.asarray(
                        packet["audio_pcm"],
                        dtype=np.float32,
                    ).reshape(-1)
                    packet_rms = (
                        float(np.sqrt(np.mean(np.square(audio_pcm, dtype=np.float32))))
                        if audio_pcm.size else 0.0
                    )
                    # Forensic fix (logs_3, 2026-09-06): playback_state["active"]
                    # feeds the mic-overlap barge-in trigger below (receive loop:
                    # mic_rms >= 0.020 while "active" => playback_interrupt_event
                    # fires, media resets, and the suppress/resume state machine
                    # takes over -- see _persona_priority_worker). The thinking
                    # sound is loud (it must be, to be audible) but it is NOT the
                    # assistant's answer, so it must never look like "assistant
                    # speaking" to that detector -- treating it as such let a
                    # multi-second search turn's own waiting cue trip a false
                    # barge-in, whose recovery condition (silence, then a fresh
                    # loud reply) is exactly what search/injection makes hard to
                    # satisfy, this being how the mute in logs_3 got triggered.
                    effective_rms = 0.0 if packet.get("masked") else packet_rms
                    if effective_rms >= 0.006:
                        playback_state["active"] = True
                        playback_state["hold"] = 4
                    elif int(playback_state["hold"]) > 0:
                        playback_state["hold"] = int(playback_state["hold"]) - 1
                        playback_state["active"] = True
                    else:
                        playback_state["active"] = False
                    opus_payload = opus_writer.append_pcm(audio_pcm)
                    if opus_payload is None and hasattr(opus_writer, "read_bytes"):
                        opus_payload = opus_writer.read_bytes()
                    if opus_payload:
                        try:
                            async with ws_send_lock:
                                await ws.send_bytes(b"\x01" + opus_payload)
                        except (WebSocketDisconnect, RuntimeError, Exception):
                            break
                        packets_sent += 1
                        bytes_sent += len(opus_payload)
                        # Record DELIVERY, not generation. Everything upstream
                        # (answer_complete, first_audio_generated) proves the
                        # model produced something; only this proves it was
                        # sent. Masked packets (thinking sound / post-injection
                        # cover) are excluded: they are loud, so counting them
                        # would report the waiting cue as the answer starting.
                        # Cheap, non-blocking, never raises.
                        if reply_engine is not None and not packet.get("masked"):
                            reply_engine.note_audio_streamed(packet_rms)

                    if packets_sent and packets_sent % 50 == 0:
                        print(
                            f"[NATIVE-AUDIO] sent={packets_sent} seq={idx} "
                            f"audio_q={audio_q.qsize()} max_q={max_queue_depth} "
                            f"interval={interval_ms:.1f}ms "
                            f"late={lateness_ms:.1f}ms max_late={max_lateness_ms:.1f}ms "
                            f"bytes={bytes_sent}",
                            flush=True,
                        )

            mic_ingest_thread = threading.Thread(
                target=_mic_ingest_worker,
                daemon=True,
                name="mic-ingest",
            )
            mic_ingest_thread.start()
            persona_thread = threading.Thread(
                target=_persona_priority_worker,
                daemon=True,
                name="persona-priority",
            )
            persona_thread.start()
            gpu_thread = threading.Thread(target=_gpu_producer_thread, daemon=True, name="gpu-producer")
            gpu_thread.start()
            print(
                f"[AJ][AUDIO] session={session_id[:8]} "
                f"video_path=/ws/video?session_id={session_id}",
                flush=True,
            )
            audio_sender_task = asyncio.create_task(_audio_sender())

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is not None:
                    if reply_engine is None:
                        continue
                    assert mic_q is not None
                    packet_sr = int(browser_input_sr)
                    if len(data) > 0 and data[0] == 1:
                        decoded_pcm = opus_reader.append_bytes(bytes(data[1:]))
                        if decoded_pcm.shape[-1] == 0:
                            continue
                        decoded_pcm = np.asarray(decoded_pcm, dtype=np.float32).reshape(-1)
                        decoded_i16 = (
                            np.clip(decoded_pcm, -1.0, 1.0) * 32767.0
                        ).astype(np.int16)
                        data = decoded_i16.tobytes()
                        packet_sr = TARGET_SR
                    audio_packets_seen += 1
                    pcm_i16 = np.frombuffer(data, dtype=np.int16)
                    mic_rms = float(np.sqrt(np.mean((pcm_i16.astype(np.float32) / 32768.0) ** 2))) if pcm_i16.size else 0.0
                    mic_peak = float(np.max(np.abs(pcm_i16.astype(np.float32) / 32768.0))) if pcm_i16.size else 0.0
                    if bool(playback_state["active"]) and mic_rms >= 0.020:
                        mic_overlap_packets += 1
                    else:
                        mic_overlap_packets = 0
                    if mic_overlap_packets >= 2 and not playback_interrupt_event.is_set():
                        playback_interrupt_event.set()
                        mic_overlap_packets = 0
                        print(
                            f"[PLAYBACK-OVERLAP] packet={audio_packets_seen} "
                            f"mic_rms={mic_rms:.5f} assistant_playback=active",
                            flush=True,
                        )
                    now_wall = time.perf_counter()
                    if audio_packets_seen <= 3 or now_wall - last_mic_level_log_wall >= 1.0:
                        voice = "VOICE" if mic_rms >= 0.02 else "quiet"
                        print(
                            f"[MIC] packet={audio_packets_seen} rms={mic_rms:.5f} "
                            f"peak={mic_peak:.3f} sr={packet_sr} {voice}",
                            flush=True,
                        )
                        last_mic_level_log_wall = now_wall
                    if audio_packets_seen <= 3:
                        print(
                            f"[liveTryHeliumFM] rx binary mic packet#{audio_packets_seen} "
                            f"bytes={len(data)} sr={packet_sr}",
                            flush=True,
                        )
                    if not session_started.is_set():
                        session_started.set()
                        print(
                            "[liveTryHeliumFM] auto-started session from first binary mic packet",
                            flush=True,
                        )
                    try:
                        mic_q.put_nowait(
                            (bytes(data), int(packet_sr), audio_packets_seen, time.perf_counter())
                        )
                    except queue.Full:
                        # The queue is intentionally unbounded. If this ever
                        # triggers, fail loudly rather than losing speech invisibly.
                        print(
                            f"[MIC-NODROP] ERROR unexpected queue.Full "
                            f"packet={audio_packets_seen}",
                            flush=True,
                        )
                    continue
                text = msg.get("text")
                if text is None:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                msg_type = str(payload.get("type", "")).lower()
                print(f"[liveTryHeliumFM] rx text type={msg_type or '<empty>'}", flush=True)

                if msg_type == "start":
                    session_started.clear()
                    playback_interrupt_event.clear()
                    playback_state["active"] = False
                    playback_state["hold"] = 0
                    mic_overlap_packets = 0
                    browser_input_sr = int(payload.get("sample_rate", payload.get("sampleRate", browser_input_sr)))
                    # A Start message is a hard session boundary. Remove any
                    # transport data that arrived before Start and recreate the
                    # stateful Opus decoder before resetting model/avatar state.
                    cleared_mic_packets = 0
                    if mic_q is not None:
                        while True:
                            try:
                                stale_item = mic_q.get_nowait()
                            except queue.Empty:
                                break
                            if stale_item is not None:
                                cleared_mic_packets += 1
                    cleared_persona_events = 0
                    while True:
                        try:
                            persona_event_q.get_nowait()
                            cleared_persona_events += 1
                        except queue.Empty:
                            break
                    opus_reader = sphn.OpusStreamReader(TARGET_SR)
                    fm_engine.reset_session()
                    if seedvc is not None:
                        seedvc.reset()
                    if fm_engine.audio_pcm is not None and (stream_task is None or stream_task.done()):
                        stream_task = asyncio.create_task(stream_from_file(ws, fm_engine))
                    if reply_engine is not None:
                        with reply_input_lock:
                            reply_engine.reset_session()
                        session_started.set()
                    print(
                        f"[SESSION-FULL-RESET] session={session_id[:8]} "
                        f"cleared_mic_packets={cleared_mic_packets} "
                        f"cleared_persona_events={cleared_persona_events} "
                        "opus=reset persona=reset avatar=reset",
                        flush=True,
                    )
                    print(
                        "[liveTryHeliumFM] start → "
                        + ("streaming from file" if fm_engine.audio_pcm is not None else "live Moshi reply mode"),
                        flush=True,
                    )

                elif msg_type == "chunk_audio":
                    pcm_b64 = payload.get("pcm_s16le_b64", "")
                    if not pcm_b64:
                        continue
                    pcm_bytes = base64.b64decode(pcm_b64)
                    result = fm_engine.feed_pcm(pcm_bytes)
                    if result is not None:
                        motion, fm_info, _pcm_chunk = result
                        avatar_chunk_id = len(fm_engine._session_chunk_rows)
                        n_frames = int(motion.shape[0])
                        emitted = fm_info["abs_start"]
                        for sb_start in range(0, n_frames, fm_engine.render_sub_batch):
                            sub = motion[sb_start:sb_start + fm_engine.render_sub_batch].to(
                                fm_engine.device, dtype=fm_engine.dtype
                            )
                            frames_np, _ = fm_engine._render_motion(sub)
                            for j, frame_rgb in enumerate(frames_np):
                                idx = emitted + sb_start + j
                                await ws.send_json({
                                    "type": "chunk_frame",
                                    "chunk_id": idx + 1,
                                    "frame_idx": 0,
                                    "jpeg_b64": encode_jpeg_b64(frame_rgb, fm_engine.jpeg_quality),
                                    "moshi_text": (
                                        f"live Helium+FM "
                                        f"helium={fm_info['helium_ms']:.0f}ms "
                                        f"fm={fm_info['fm_ms']:.0f}ms"
                                    ),
                                    "server_fps": round(float(args.fps), 1),
                                    "chunks_done": avatar_chunk_id,
                                })

                elif msg_type == "stop":
                    print("[liveTryHeliumFM] stop requested", flush=True)
                    break

        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            session = split_sessions.get(session_id)
            if session is not None:
                session["closed"] = True
            # Signal every worker before joining it. Previously teardown joined
            # workers whose loops were still active, so a rapid second Start
            # waited forever and stale PersonaPlex output survived the boundary.
            if reply_engine is not None:
                pipeline_stop_event.set()
                session_started.clear()
            if stream_task is not None:
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            if mic_q is not None:
                with contextlib.suppress(queue.Full):
                    mic_q.put_nowait(None)
            if mic_ingest_thread is not None:
                await asyncio.to_thread(mic_ingest_thread.join, 5.0)
            if persona_thread is not None:
                await asyncio.to_thread(persona_thread.join, 10.0)
            if gpu_thread is not None:
                await asyncio.to_thread(gpu_thread.join, 30.0)
            if sender_task is not None:
                sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await sender_task
            if audio_sender_task is not None:
                audio_sender_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await audio_sender_task
            fm_engine.dump_last_session(source="websocket_live")
            split_sessions.pop(session_id, None)
            if conversation_lock.locked():
                conversation_lock.release()
                print("[SESSION-STRICT] released PersonaPlex session lock", flush=True)
            print("[liveTryHeliumFM] websocket closed", flush=True)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    syslog = runtime_logging.get_system_logger()
    runtime_logging.log_event(
        syslog, "Server", "startup_begin",
        logs_dir=str(runtime_logging.logs_dir()),
        torch_version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda,
    )
    parser = LiveHeliumFMOptions()
    args = parser.parse()
    args.rank = args.device
    parser.print_options()

    app = build_app(args)

    import uvicorn

    print(f"[liveTryHeliumFM_ws_binary] serving {args.html_path} (binary av_transport)")
    print(f"[liveTryHeliumFM] open http://{args.host}:{args.port}/")
    runtime_logging.log_event(
        syslog, "Server", "listening",
        host=args.host, port=args.port, html_path=args.html_path,
        search_enabled=bool(getattr(args, "stt_hf_repo", "") and getattr(args, "compressor_model", "")),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
