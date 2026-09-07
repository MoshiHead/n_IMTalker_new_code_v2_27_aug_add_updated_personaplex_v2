"""liveTry.py - v3 one-websocket server, Step 3: Moshi audio/text only.

What this version does:
    browser mic PCM -> /ws/conversation -> original Moshi/Mimi
    Moshi reply audio/text -> JSON chunk_audio + static JPEG chunk_frame

What this version deliberately does NOT do yet:
    no VAD
    no Helium extraction
    no FM
    no IMTalker renderer
    no WebRTC / TURN / H264 / Opus

The goal is to prove the teammate-style HTML protocol works cleanly with our
original Moshi backend before adding Helium/IMTalker.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
from dataclasses import fields, is_dataclass
import json
import logging
import os
import sys
import tarfile
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

import runtime_logging  # IMTalker/ is on sys.path by the time this runs


TARGET_SR = 24000
FRAME_SIZE = 1920  # 80 ms at 24 kHz, Moshi/Mimi step size


def _ensure_moshi_importable(moshi_root: str | Path) -> None:
    root = Path(moshi_root)
    pkg = root / "moshi"
    if pkg.exists() and str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))


def _clean_text_piece(piece: str) -> str:
    return piece.replace("▁", " ")


def _wrap_personaplex_system_prompt(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


_STREAMING_STATE_SKIP = object()


def _clone_streaming_state_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        cloned = {}
        for field in fields(value):
            item = _clone_streaming_state_value(getattr(value, field.name))
            if item is not _STREAMING_STATE_SKIP:
                cloned[field.name] = item
        return cloned
    if isinstance(value, dict):
        cloned = {}
        for key, item_value in value.items():
            item = _clone_streaming_state_value(item_value)
            if item is not _STREAMING_STATE_SKIP:
                cloned[key] = item
        return cloned
    if hasattr(value, "asdict"):
        return _clone_streaming_state_value(value.asdict())
    return _STREAMING_STATE_SKIP


def _restore_streaming_state_value(target, saved) -> None:
    if isinstance(target, torch.Tensor):
        if isinstance(saved, torch.Tensor):
            target.copy_(saved.to(device=target.device, dtype=target.dtype))
        else:
            target.copy_(torch.as_tensor(saved, device=target.device, dtype=target.dtype))
        return
    if is_dataclass(target) and not isinstance(target, type):
        for name, item in saved.items():
            current = getattr(target, name)
            if isinstance(current, torch.Tensor) or is_dataclass(current) or isinstance(current, dict) or hasattr(current, "asdict"):
                _restore_streaming_state_value(current, item)
            else:
                setattr(target, name, item)
        return
    if isinstance(target, dict):
        for name, item in saved.items():
            current = target[name]
            if isinstance(current, torch.Tensor) or is_dataclass(current) or isinstance(current, dict) or hasattr(current, "asdict"):
                _restore_streaming_state_value(current, item)
            else:
                target[name] = item


def _snapshot_streaming_state(module) -> dict:
    snapshot = {}
    for name, state in module.get_streaming_state().items():
        cloned = _clone_streaming_state_value(state)
        if cloned is not _STREAMING_STATE_SKIP:
            snapshot[name] = cloned
    return snapshot


def _restore_streaming_state(module, snapshot: dict) -> None:
    current = module.get_streaming_state()
    for name, saved in snapshot.items():
        if name in current:
            _restore_streaming_state_value(current[name], saved)


def _make_placeholder_jpeg(path: str | Path | None) -> str:
    img = None
    if path:
        p = Path(path)
        if p.is_file():
            img = cv2.imread(str(p))
    if img is None:
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        cv2.putText(
            img,
            "Moshi",
            (150, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (235, 235, 235),
            3,
            cv2.LINE_AA,
        )
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok:
        raise RuntimeError("failed to encode placeholder JPEG")
    return base64.b64encode(enc.tobytes()).decode("ascii")


class MoshiOnlyEngine:
    def __init__(
        self,
        *,
        moshi_root: str,
        mimi_hf_repo: str,
        device: str,
        cfg_coef: float,
        placeholder_jpeg_b64: str,
        moshi_weight: str = "",
        mimi_weight: str = "",
        tokenizer: str = "",
        quantize_4bit: bool = False,
        num_codebooks: int = 8,
        context: int | None = None,
        voice_prompt: str = "",
        voice_prompt_dir: str = "",
        text_prompt: str = "",
        # --- STT + query routing + web search (all optional, off by default) ---
        ref_lora_dir: str = "",
        merge_ref_lora: bool = False,
        max_ref_tokens: int = 250,
        stt_hf_repo: str = "",
        stt_pkg_dir: str = "",
        vad_threshold: float = 0.5,
        suppress_text_during_search: bool = True,
        stt_reject_foreign_script: bool = True,
        stt_max_non_latin_ratio: float = 0.15,
        stt_require_english: bool = True,
        compressor_model: str = "",
        compressor_device: str = "cuda",
        compressor_4bit: bool = True,
        compressor_max_passages: int = 2,
        router_threshold: float = 0.40,
        router_use_rules: bool = True,
        web_search_enabled: bool = False,
        web_search_api_key: str | None = None,
        web_search_provider: str = "tavily",
        web_search_max_results: int = 3,
        web_search_timeout: float = 3.0,
        web_search_min_score: float = 0.15,
        conversation_log_dir: str = "",
    ) -> None:
        from conversation_logger import ConversationLogger  # IMTalker/ is on sys.path by the time this runs

        self.conv_logger = ConversationLogger(log_dir=conversation_log_dir)

        _ensure_moshi_importable(moshi_root)
        from moshi.models import LMGen, loaders

        self.device = torch.device(device)
        self.placeholder_jpeg_b64 = placeholder_jpeg_b64
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.step = 0
        self.skip_first = True
        self.sampled_text = ""
        self.audio_text = ""
        self.started_at = time.perf_counter()
        self.text_prompt = str(text_prompt or "")
        self.voice_prompt = str(voice_prompt or "")
        self.voice_prompt_dir = str(voice_prompt_dir or "")
        self._hf_repo = mimi_hf_repo
        self.suppress_text_during_search = bool(suppress_text_during_search)
        self._prompt_state_cache_key = None
        self._prompt_state_cache = None
        self._prompt_state_cache_hits = 0
        self._prompt_state_cache_pending_key = None
        self._use_prompt_state_cache = (
            os.environ.get("IMTALKER_PROMPT_STATE_CACHE", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )

        print(
            "[liveTry] loading Moshi "
            f"repo={mimi_hf_repo} root={moshi_root} device={self.device} cfg={cfg_coef}"
        )
        syslog = runtime_logging.get_system_logger()
        runtime_logging.log_event(
            syslog, "PersonaPlex", "load_start",
            source=mimi_hf_repo, moshi_root=moshi_root, device=str(self.device),
            dtype="bfloat16", quantize_4bit=bool(quantize_4bit),
        )
        t0 = time.perf_counter()
        if hasattr(loaders, "CheckpointInfo"):
            ckpt_info = loaders.CheckpointInfo.from_hf_repo(mimi_hf_repo)
            self.mimi = ckpt_info.get_mimi(device=self.device)
            self.lm = ckpt_info.get_moshi(device=self.device, dtype=torch.bfloat16)
            self.tokenizer = ckpt_info.get_text_tokenizer()
            model_type = getattr(ckpt_info, "model_type", "moshi")
        else:
            from huggingface_hub import hf_hub_download
            import inspect
            import sentencepiece

            repo = mimi_hf_repo or getattr(loaders, "DEFAULT_REPO", "nvidia/personaplex-7b-v1")
            if not mimi_weight:
                mimi_weight = hf_hub_download(repo, loaders.MIMI_NAME)
            if not moshi_weight:
                moshi_weight = hf_hub_download(repo, loaders.MOSHI_NAME)
            if not tokenizer:
                tokenizer = hf_hub_download(repo, loaders.TEXT_TOKENIZER_NAME)
            self.mimi = loaders.get_mimi(mimi_weight, self.device)
            lm_kwargs = {"device": self.device, "dtype": torch.bfloat16}
            supported = set(inspect.signature(loaders.get_moshi_lm).parameters)
            optional_kwargs = {
                "quantize_4bit": bool(quantize_4bit),
                "num_codebooks": int(num_codebooks),
                "context": context,
            }
            lm_kwargs.update({k: v for k, v in optional_kwargs.items() if k in supported})
            self.lm = loaders.get_moshi_lm(moshi_weight, **lm_kwargs)
            self.tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)  # type: ignore
            model_type = "personaplex"

        runtime_logging.log_event(
            syslog, "PersonaPlex", "load_done",
            source=mimi_hf_repo, model_type=model_type, device=str(self.device),
            lm_dtype=str(next(self.lm.parameters()).dtype), quantize_4bit=bool(quantize_4bit),
            moshi_weight=moshi_weight or "(from repo)", mimi_weight=mimi_weight or "(from repo)",
            tokenizer_path=tokenizer or "(from repo)",
            duration_ms=round((time.perf_counter() - t0) * 1000.0, 1),
        )

        # The reference LoRA must be applied here: after the base LM is loaded,
        # before LMGen(...)/CUDA-graph capture below (and before the prompt
        # state cache is captured post-warmup) -- PEFT mutates self.lm's
        # submodules in place, so both the graph and the cached prompt state
        # bake in the LoRA-augmented forward pass either way. This adapter
        # teaches the model to consume the injected <lookup>/<ref> tags that
        # carry web-search context.
        self.ref_lora_dir = str(ref_lora_dir or "")
        if self.ref_lora_dir:
            self._load_ref_lora(self.ref_lora_dir, merge_lora=bool(merge_ref_lora))

        self.mimi.eval()
        self.lm.eval()

        try:
            from moshi.run_inference import get_condition_tensors

            cond_tensors = get_condition_tensors(
                model_type,
                self.lm,
                batch_size=1,
                cfg_coef=float(cfg_coef),
            )
        except Exception:
            cond_tensors = {}

        def on_text_hook(text_tokens: torch.Tensor) -> None:
            token = int(text_tokens[0].detach().item())
            piece = self.decode_piece(token)
            if piece:
                self.sampled_text += piece

        try:
            self.lm_gen = LMGen(
                self.lm,
                cfg_coef=float(cfg_coef),
                condition_tensors=cond_tensors,
                on_text_hook=on_text_hook,
            )
        except TypeError:
            self.lm_gen = LMGen(self.lm, device=self.device)
        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        if self.frame_size != FRAME_SIZE:
            raise RuntimeError(f"expected Mimi frame_size={FRAME_SIZE}, got {self.frame_size}")
        self._warmup_runtime()
        if self._use_prompt_state_cache:
            self._capture_prompt_state_cache()
        self.reset_session()
        print(f"[liveTry] Moshi ready in {time.perf_counter() - t0:.1f}s")

        # --- STT / query router / web search: all optional, each independently
        # try/except-guarded so a failure here never blocks avatar startup. ---
        self.max_ref_tokens = int(max_ref_tokens)
        self.vad_threshold = float(vad_threshold)
        self.stt_reject_foreign_script = bool(stt_reject_foreign_script)
        self.stt_max_non_latin_ratio = float(stt_max_non_latin_ratio)
        self.stt_require_english = bool(stt_require_english)
        self.web_search_enabled = bool(web_search_enabled)
        self.web_search_api_key = web_search_api_key or None
        self.web_search_provider = str(web_search_provider)
        self.web_search_max_results = int(web_search_max_results)
        self.web_search_timeout = float(web_search_timeout)
        self.web_search_min_score = float(web_search_min_score)
        if self.web_search_enabled and not self.web_search_api_key:
            print(
                "[liveTry] web_search_enabled but no web_search_api_key configured "
                "-- web search will no-op at request time",
                flush=True,
            )

        # STT is the sole source of the transcript the router reads.
        self.stt_mimi = None
        self.stt_lm_gen = None
        self.stt_tokenizer = None
        self.stt_padding_token_id = 3
        if stt_hf_repo and stt_pkg_dir:
            self._load_stt_vad(str(stt_hf_repo), str(stt_pkg_dir), self.device)

        # The compressor's small instruct model does double duty: it both
        # compresses web results into one speakable sentence AND backs the
        # query router (one shared model, one load, no extra VRAM -- see
        # QueryRouter.from_compressor). The router therefore requires the
        # compressor; if the compressor fails to load, routing is unavailable
        # and every turn is answered from the model's own knowledge.
        self.context_compressor = None
        self.query_router = None
        if self.stt_lm_gen is not None and compressor_model:
            self._load_context_compressor(
                str(compressor_model), str(compressor_device), bool(compressor_4bit), int(compressor_max_passages)
            )
            if self.context_compressor is not None:
                self._load_query_router(float(router_threshold), bool(router_use_rules))

        # True only when a transcript can be produced AND routed. When False,
        # the avatar behaves exactly like the plain conversational server: no
        # transcription side-effects, no injection, no search.
        self.search_enabled = self.stt_lm_gen is not None and self.query_router is not None
        if self.stt_lm_gen is not None and self.query_router is None:
            print(
                "[liveTry] STT loaded but no query router -- web search disabled; "
                "every turn will be answered from the model's own knowledge",
                flush=True,
            )

        self.conv_logger.component_status(
            ref_lora_loaded=bool(self.ref_lora_dir),
            stt_loaded=self.stt_lm_gen is not None,
            compressor_loaded=self.context_compressor is not None,
            router_loaded=self.query_router is not None,
            search_enabled=self.search_enabled,
            web_search_enabled=self.web_search_enabled,
            web_search_has_key=bool(self.web_search_api_key),
        )

    def _load_ref_lora(self, checkpoint_dir: str, merge_lora: bool = False) -> None:
        """Load the <lookup>/<ref> LoRA adapter onto self.lm. Unmerged by
        default (QLoRA-style: LoRA computed at forward time on top of the 4-bit
        base, not merged into the quantized weights). PEFT mutates self.lm's
        target submodules in place -- self.lm keeps pointing at the same, now
        LoRA-augmented, object either way."""
        syslog = runtime_logging.get_system_logger()
        lora_path = Path(checkpoint_dir) / "lora"
        if not lora_path.exists():
            print(f"[liveTry] no lora/ at {lora_path} -- skipping", flush=True)
            runtime_logging.log_event(
                syslog, "RefLoRA", "not_found", level=logging.WARNING, path=str(lora_path),
            )
            return
        t0 = time.perf_counter()
        try:
            from peft import PeftModel

            print(f"[liveTry] loading reference LoRA from {lora_path} (merge={merge_lora})", flush=True)
            peft_model = PeftModel.from_pretrained(self.lm, str(lora_path))
            if merge_lora:
                self.lm = peft_model.merge_and_unload()
            print(f"[liveTry] reference LoRA loaded from {lora_path}", flush=True)
            runtime_logging.log_event(
                syslog, "RefLoRA", "loaded",
                name="helium_lora_v1 (<lookup>/<ref> adapter)", path=str(lora_path),
                base_model="PersonaPlex LM", device=str(self.device),
                merged=bool(merge_lora), status="ok",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[liveTry] reference LoRA load failed (continuing without it): {e!r}\n{tb}", flush=True)
            self.conv_logger.error("ref_lora_load", e, tb)
            runtime_logging.log_event(
                syslog, "RefLoRA", "load_failed", level=logging.ERROR, path=str(lora_path), error=repr(e),
            )

    def _load_stt_vad(self, stt_hf_repo: str, stt_pkg_dir: str, device) -> None:
        # Import search_helpers first, on its own, with a specific error
        # message -- a missing IMTalker/search_helpers.py on the deployed
        # checkout is the single most likely cause of a silent "search
        # disabled" (it's a plain ModuleNotFoundError that the broad except
        # below would otherwise bury under a generic message).
        syslog = runtime_logging.get_system_logger()
        try:
            import search_helpers
        except ImportError as e:
            tb = traceback.format_exc()
            print(
                f"[liveTry] search disabled -- could not import search_helpers ({e!r}). "
                f"Check that IMTalker/search_helpers.py exists in this checkout (a missing file "
                f"here is the most common cause of search silently not loading).\n{tb}",
                flush=True,
            )
            self.conv_logger.error("search_helpers_import", e, tb)
            runtime_logging.log_event(syslog, "STT", "load_failed", level=logging.ERROR, error=repr(e))
            return

        t0 = time.perf_counter()
        try:
            print(f"[liveTry] loading STT model from {stt_hf_repo}...", flush=True)
            runtime_logging.log_event(
                syslog, "STT", "load_start", source=stt_hf_repo, pkg_dir=stt_pkg_dir, device=str(device),
            )
            moshi_stt = search_helpers.load_upstream_moshi_stt(stt_pkg_dir)
            stt_info = moshi_stt.models.loaders.CheckpointInfo.from_hf_repo(stt_hf_repo)
            self.stt_mimi = stt_info.get_mimi(device=device)
            stt_lm = stt_info.get_moshi(device=device, dtype=torch.bfloat16)
            stt_lm.eval()
            self.stt_lm_gen = moshi_stt.models.LMGen(stt_lm, temp=0, temp_text=0.0)
            self.stt_lm_gen.streaming_forever(1)
            self.stt_mimi.streaming_forever(1)
            self.stt_tokenizer = stt_info.get_text_tokenizer()
            self.stt_padding_token_id = stt_info.raw_config.get("text_padding_token_id", 3)
            # Print the identity of BOTH tokenizers -- see IMTalker/liveTry.py
            # for why a matching vocab size here is a red flag.
            stt_desc = search_helpers.describe_tokenizer(self.stt_tokenizer)
            main_desc = search_helpers.describe_tokenizer(self.tokenizer)
            print(
                f"[liveTry] STT model loaded: "
                f"params={sum(p.numel() for p in stt_lm.parameters()) / 1e9:.1f}B "
                f"padding_id={self.stt_padding_token_id}",
                flush=True,
            )
            print(f"[liveTry] stt tokenizer={stt_desc}  |  main tokenizer={main_desc}", flush=True)
            if stt_desc == main_desc:
                print(
                    "[liveTry] WARNING the STT and PersonaPlex tokenizers look identical. "
                    "The STT text stream may be decoded with the wrong vocabulary, which shows up "
                    "as transcripts in a language the en/fr STT model cannot actually produce.",
                    flush=True,
                )
            self.conv_logger.event(
                "stt_loaded", stt_repo=stt_hf_repo, stt_tokenizer=stt_desc,
                main_tokenizer=main_desc, padding_token_id=self.stt_padding_token_id,
            )
            runtime_logging.log_event(
                syslog, "STT", "load_done", source=stt_hf_repo, device=str(device),
                params_b=round(sum(p.numel() for p in stt_lm.parameters()) / 1e9, 2),
                dtype="bfloat16", status="ok",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"[liveTry] search disabled -- STT model load failed: {e!r}\n{tb}", flush=True)
            self.conv_logger.error("stt_load", e, tb)
            runtime_logging.log_event(
                syslog, "STT", "load_failed", level=logging.ERROR, source=stt_hf_repo, error=repr(e),
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            )
            self.stt_mimi = None
            self.stt_lm_gen = None

    def _load_context_compressor(
        self, compressor_model: str, compressor_device: str, compressor_4bit: bool, compressor_max_passages: int
    ) -> None:
        syslog = runtime_logging.get_system_logger()
        t0 = time.perf_counter()
        runtime_logging.log_event(
            syslog, "Compressor", "load_start", source=compressor_model,
            device=compressor_device, quantize_4bit=bool(compressor_4bit),
        )
        try:
            import search_helpers

            self.context_compressor = search_helpers.ContextCompressor(
                model_name=compressor_model,
                device=compressor_device,
                quantize_4bit=compressor_4bit,
                max_passages=compressor_max_passages,
            )
            runtime_logging.log_event(
                syslog, "Compressor", "load_done", source=compressor_model,
                device=compressor_device, quantize_4bit=bool(compressor_4bit), status="ok",
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(
                f"[liveTry] context compressor disabled -- load failed: {e!r} "
                f"(web search will be disabled too: the router shares this model)\n{tb}",
                flush=True,
            )
            self.conv_logger.error("compressor_load", e, tb)
            runtime_logging.log_event(
                syslog, "Compressor", "load_failed", level=logging.ERROR,
                source=compressor_model, error=repr(e),
                duration_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            )
            self.context_compressor = None

    def _load_query_router(self, threshold: float, use_rules: bool) -> None:
        """Build the search/no-search router on top of the already-loaded
        compressor model. Shares weights, so this costs no extra VRAM and no
        extra load time -- only the Yes/No token-id lookup."""
        syslog = runtime_logging.get_system_logger()
        try:
            import search_helpers

            self.query_router = search_helpers.QueryRouter.from_compressor(
                self.context_compressor, threshold=threshold, use_rules=use_rules,
            )
            print(
                f"[liveTry] query router ready (threshold={threshold:.2f} "
                f"rules={'on' if use_rules else 'off'}, sharing the compressor model)",
                flush=True,
            )
            runtime_logging.log_event(
                syslog, "QueryRouter", "ready", threshold=threshold, rules=bool(use_rules), status="ok",
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(
                f"[liveTry] query router disabled -- init failed: {e!r} "
                f"(every turn will be answered from the model's own knowledge)\n{tb}",
                flush=True,
            )
            self.conv_logger.error("router_load", e, tb)
            runtime_logging.log_event(
                syslog, "QueryRouter", "load_failed", level=logging.ERROR, error=repr(e),
            )
            self.query_router = None

    def _resolve_voice_prompt_path(self) -> str:
        if not self.voice_prompt:
            return ""
        if os.path.isabs(self.voice_prompt) and os.path.exists(self.voice_prompt):
            return self.voice_prompt
        if self.voice_prompt_dir and os.path.isdir(self.voice_prompt_dir):
            candidate = os.path.join(self.voice_prompt_dir, self.voice_prompt)
            if os.path.exists(candidate):
                return candidate
        from huggingface_hub import hf_hub_download

        voices_tgz = Path(hf_hub_download(self._hf_repo, "voices.tgz"))
        voices_dir = voices_tgz.parent / "voices"
        if not voices_dir.exists():
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(path=voices_tgz.parent)
        candidate = voices_dir / self.voice_prompt
        if not candidate.exists():
            raise FileNotFoundError(f"voice prompt not found: {candidate}")
        self.voice_prompt_dir = str(voices_dir)
        return str(candidate)

    @torch.no_grad()
    def _apply_system_prompts(self) -> None:
        if not hasattr(self.lm_gen, "step_system_prompts"):
            return
        voice_path = self._resolve_voice_prompt_path()
        raw_voice_prompt = bool(voice_path and not voice_path.endswith(".pt"))
        if voice_path and getattr(self.lm_gen, "voice_prompt", None) != voice_path:
            if voice_path.endswith(".pt") and hasattr(self.lm_gen, "load_voice_prompt_embeddings"):
                self.lm_gen.load_voice_prompt_embeddings(voice_path)
            elif hasattr(self.lm_gen, "load_voice_prompt"):
                self.lm_gen.load_voice_prompt(voice_path)
            print(f"[liveTry] voice prompt: {voice_path}", flush=True)
        elif voice_path:
            print(f"[liveTry] voice prompt reused: {voice_path}", flush=True)
        wrapped_prompt = ""
        if self.text_prompt and hasattr(self.tokenizer, "encode"):
            with contextlib.suppress(Exception):
                wrapped_prompt = _wrap_personaplex_system_prompt(self.text_prompt)
                self.lm_gen.text_prompt_tokens = self.tokenizer.encode(wrapped_prompt)
                print(
                    "[liveTry] text prompt loaded "
                    f"tokens={len(self.lm_gen.text_prompt_tokens)} "
                    f"wrapper=personaplex text={self.text_prompt[:80]!r}",
                    flush=True,
                )
        else:
            self.lm_gen.text_prompt_tokens = None
        cache_key = (
            voice_path,
            wrapped_prompt,
            int(getattr(self.lm_gen, "audio_silence_frame_cnt", -1)),
        )
        if (
            self._use_prompt_state_cache
            and self._prompt_state_cache_key == cache_key
            and self._prompt_state_cache is not None
        ):
            self.lm_gen.set_streaming_state_inplace(dict(self._prompt_state_cache))
            self._prompt_state_cache_hits += 1
            print(
                "[liveTry] restored post-warmup native LM streaming state cache "
                f"hit={self._prompt_state_cache_hits}",
                flush=True,
            )
            return
        encoder_graph = None
        if raw_voice_prompt:
            state = getattr(self.mimi, "_streaming_state", None)
            encoder_graph = getattr(state, "graphed_tr_enc", None)
            if encoder_graph is not None:
                encoder_graph.disable = True
        try:
            self.lm_gen.step_system_prompts(self.mimi)
            if self._use_prompt_state_cache:
                self._prompt_state_cache_pending_key = cache_key
                print(
                    "[liveTry] replayed prompts normally; cache capture waits for warmup",
                    flush=True,
                )
            else:
                print(
                    "[liveTry] prompt state cache disabled; replayed prompts normally",
                    flush=True,
                )
        finally:
            with contextlib.suppress(Exception):
                self.mimi.reset_streaming()
            if encoder_graph is not None:
                encoder_graph.reset()
                encoder_graph.disable = False

    def _capture_prompt_state_cache(self) -> None:
        if self._prompt_state_cache_pending_key is None:
            raise RuntimeError("cannot capture prompt cache before prompts are applied")
        from moshi.modules.streaming import _flatten_streaming_state

        tensors = {}
        metadata = {}
        _flatten_streaming_state(
            tensors,
            metadata,
            self.lm_gen.get_streaming_state(),
            prefix="",
        )
        cache = {
            name: value.detach().cpu().clone()
            for name, value in tensors.items()
        }
        cache.update(metadata)
        self._prompt_state_cache = cache
        self._prompt_state_cache_key = self._prompt_state_cache_pending_key
        print(
            "[liveTry] captured post-warmup native LM streaming state cache "
            f"tensors={len(tensors)} metadata={len(metadata)}",
            flush=True,
        )

    def reset_session(self) -> None:
        self.input_buffer = np.zeros(0, dtype=np.float32)
        self.step = 0
        self.skip_first = True
        self.sampled_text = ""
        self.audio_text = ""
        self.started_at = time.perf_counter()
        with contextlib.suppress(Exception):
            self.mimi.reset_streaming()
        with contextlib.suppress(Exception):
            self.lm_gen.reset_streaming()
        # Guarded with getattr: reset_session() is also called from inside
        # _warmup_runtime(), before the STT submodel (loaded at the very end
        # of __init__, after warmup) exists yet.
        stt_lm_gen = getattr(self, "stt_lm_gen", None)
        if stt_lm_gen is not None:
            with contextlib.suppress(Exception):
                stt_lm_gen.reset_streaming()
            with contextlib.suppress(Exception):
                self.stt_mimi.reset_streaming()
        self._apply_system_prompts()

    @torch.no_grad()
    def _warmup_runtime(self, n_steps: int = 6) -> None:
        t0 = time.perf_counter()
        silence = torch.zeros(1, 1, self.frame_size, device=self.device, dtype=torch.float32)
        for idx in range(int(n_steps)):
            codes = self.mimi.encode(silence)
            if idx == 0:
                self.mimi.reset_streaming()
            tokens = self.lm_gen.step(codes[:, :, :1])
            if tokens is not None:
                reply = self.mimi.decode(tokens[:, 1:])
                _ = reply.detach().float().mean().item()
        self.reset_session()
        _sync = getattr(torch.cuda, "synchronize", None)
        if callable(_sync) and torch.cuda.is_available():
            _sync()
        print(f"[liveTry] Moshi runtime warmup done in {1000.0 * (time.perf_counter() - t0):.0f}ms")

    def decode_piece(self, token: int) -> str:
        if token in (0, 3):
            return ""
        with contextlib.suppress(Exception):
            return _clean_text_piece(self.tokenizer.id_to_piece(int(token)))
        return ""

    def append_browser_pcm(self, pcm_i16: np.ndarray, input_sr: int) -> None:
        pcm = pcm_i16.astype(np.float32) / 32768.0
        if int(input_sr) != TARGET_SR:
            wav = torch.from_numpy(pcm).view(1, -1)
            pcm = torchaudio.functional.resample(wav, int(input_sr), TARGET_SR)[0].numpy()
        self.input_buffer = np.concatenate([self.input_buffer, pcm.astype(np.float32, copy=False)])

    @torch.no_grad()
    def process_ready_steps(self) -> list[dict]:
        events: list[dict] = []
        while self.input_buffer.shape[0] >= FRAME_SIZE:
            pcm = self.input_buffer[:FRAME_SIZE].copy()
            self.input_buffer = self.input_buffer[FRAME_SIZE:].copy()
            events.append(self._step(pcm))
        return events

    @torch.no_grad()
    def _step(self, pcm24: np.ndarray) -> dict:
        self.step += 1
        t0 = time.perf_counter()
        chunk = torch.from_numpy(pcm24).to(self.device, dtype=torch.float32)[None, None]

        t_encode0 = time.perf_counter()
        codes = self.mimi.encode(chunk)
        t_encode1 = time.perf_counter()
        if self.skip_first:
            # Same first-frame reset used in Moshi examples/live code.
            self.mimi.reset_streaming()
            self.skip_first = False

        t_lm0 = time.perf_counter()
        tokens = self.lm_gen.step(codes[:, :, :1])
        t_lm1 = time.perf_counter()

        token = -1
        token_piece = ""
        decode_ms = 0.0
        if tokens is None:
            reply_pcm = np.zeros(FRAME_SIZE, dtype=np.float32)
        else:
            token = int(tokens[0, 0, 0].detach().item())
            token_piece = self.decode_piece(token)
            if token_piece:
                self.audio_text += token_piece
            t_decode0 = time.perf_counter()
            reply = self.mimi.decode(tokens[:, 1:])
            reply_pcm = reply[0, 0].detach().float().cpu().numpy()
            decode_ms = 1000.0 * (time.perf_counter() - t_decode0)
            if reply_pcm.shape[0] < FRAME_SIZE:
                reply_pcm = np.pad(reply_pcm, (0, FRAME_SIZE - reply_pcm.shape[0]))
            elif reply_pcm.shape[0] > FRAME_SIZE:
                reply_pcm = reply_pcm[:FRAME_SIZE]

        reply_rms = float(np.sqrt(np.mean(np.square(reply_pcm, dtype=np.float32))))
        reply_peak = float(np.max(np.abs(reply_pcm))) if reply_pcm.size else 0.0
        input_rms = float(np.sqrt(np.mean(np.square(pcm24, dtype=np.float32))))
        encode_ms = 1000.0 * (t_encode1 - t_encode0)
        lm_ms = 1000.0 * (t_lm1 - t_lm0)
        total_ms = 1000.0 * (time.perf_counter() - t0)

        reply_i16 = np.clip(reply_pcm, -1.0, 1.0)
        reply_i16 = (reply_i16 * 32767.0).astype(np.int16)
        audio_b64 = base64.b64encode(reply_i16.tobytes()).decode("ascii")

        print(
            "[liveTry] moshi "
            f"step={self.step} token={token} piece={token_piece!r} "
            f"in_rms={input_rms:.5f} reply_rms={reply_rms:.5f} peak={reply_peak:.3f} "
            f"encode={encode_ms:.1f}ms lm={lm_ms:.1f}ms decode={decode_ms:.1f}ms total={total_ms:.1f}ms"
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
        }


def build_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="IMTalker Moshi liveTry")
    started_at = time.perf_counter()
    html_path = Path(args.html_path)
    placeholder_jpeg_b64 = _make_placeholder_jpeg(args.placeholder_path)
    engine: MoshiOnlyEngine | None = None

    def get_engine() -> MoshiOnlyEngine:
        nonlocal engine
        if engine is None:
            engine = MoshiOnlyEngine(
                moshi_root=args.moshi_root,
                mimi_hf_repo=args.mimi_hf_repo,
                device=args.device,
                cfg_coef=args.cfg_coef,
                placeholder_jpeg_b64=placeholder_jpeg_b64,
            )
        return engine

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
        return HTMLResponse(
            f"<h1>Missing HTML</h1><p>Expected: {html_path}</p>",
            status_code=500,
        )

    @app.get("/health")
    async def health():
        return JSONResponse({
            "ok": True,
            "stage": "moshi_text_audio_only",
            "uptime_sec": round(time.perf_counter() - started_at, 3),
            "moshi_loaded": engine is not None,
        })

    @app.websocket("/ws/conversation")
    async def conversation(ws: WebSocket):
        await ws.accept()
        input_sr = 48000
        packets = 0
        samples = 0
        t0 = time.perf_counter()
        moshi = get_engine()

        await ws.send_json({
            "type": "server_ready",
            "sample_rate": TARGET_SR,
            "model_type": "moshi-only",
            "tokens_per_chunk": 1,
            "buffer_ms": 400,
        })
        print("[liveTry] websocket connected; sent server_ready")

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                text = msg.get("text")
                if text is not None:
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        print(f"[liveTry] bad json: {text[:120]!r}")
                        continue

                    msg_type = str(payload.get("type", "")).lower()
                    if msg_type == "start":
                        input_sr = int(payload.get("sample_rate", payload.get("sampleRate", input_sr)))
                        print(f"[liveTry] start: browser_sample_rate={input_sr}")
                    elif msg_type == "stop":
                        print("[liveTry] stop requested")
                        break
                    else:
                        print(f"[liveTry] text message: {payload}")
                    continue

                data = msg.get("bytes")
                if not data:
                    continue
                pcm_i16 = np.frombuffer(data, dtype=np.int16)
                if pcm_i16.size == 0:
                    continue

                packets += 1
                samples += int(pcm_i16.size)
                if packets == 1 or packets % 50 == 0:
                    pcm_f32 = pcm_i16.astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(np.square(pcm_f32, dtype=np.float32))))
                    elapsed = max(time.perf_counter() - t0, 1e-6)
                    print(
                        "[liveTry] mic "
                        f"packets={packets} samples={samples} "
                        f"audio_sec={samples / max(float(input_sr), 1.0):.2f} "
                        f"wall_sec={elapsed:.2f} rms={rms:.5f}"
                    )

                moshi.append_browser_pcm(pcm_i16, input_sr)
                for ev in moshi.process_ready_steps():
                    await ws.send_json({
                        "type": "chunk_audio",
                        "chunk_id": ev["step"],
                        "sample_rate": ev["sample_rate"],
                        "pcm_s16le_b64": ev["reply_i16_b64"],
                        "gen_ms": ev["total_ms"],
                    })
                    # Two static frames per 80 ms Moshi step ~= 25 fps.
                    for frame_idx in range(2):
                        await ws.send_json({
                            "type": "chunk_frame",
                            "chunk_id": ev["step"],
                            "frame_idx": frame_idx,
                            "jpeg_b64": moshi.placeholder_jpeg_b64,
                            "server_fps": 25.0,
                            "chunks_done": ev["step"],
                            "avg_gen_ms": ev["total_ms"],
                            "moshi_text": ev["audio_text"] or ev["sampled_text"],
                        })
        except WebSocketDisconnect:
            pass
        finally:
            elapsed = max(time.perf_counter() - t0, 1e-6)
            print(
                "[liveTry] websocket closed "
                f"packets={packets} samples={samples} "
                f"audio_sec={samples / max(float(input_sr), 1.0):.2f} wall_sec={elapsed:.2f}"
            )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--html_path", default=str(Path(__file__).resolve().parent / "static" / "index_v3.html"))
    parser.add_argument("--placeholder_path", default="")
    parser.add_argument("--moshi_root", default="/workspace/moshi")
    parser.add_argument("--mimi_hf_repo", default="kyutai/moshiko-pytorch-bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cfg_coef", type=float, default=1.0)
    args = parser.parse_args()

    app = build_app(args)

    import uvicorn

    print(f"[liveTry] serving {args.html_path}")
    print(f"[liveTry] open http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
