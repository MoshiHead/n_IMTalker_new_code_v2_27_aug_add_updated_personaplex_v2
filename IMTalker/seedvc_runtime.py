from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import threading
import time

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


class SeedVCStreamingConverter:
    """Stateful Seed-VC conversion for fixed-duration PersonaPlex PCM blocks."""

    def __init__(
        self,
        repo: str,
        reference_audio: str,
        input_sample_rate: int = 24_000,
        block_seconds: float = 2.0,
        diffusion_steps: int = 6,
        cfg_rate: float = 0.7,
        prompt_seconds: float = 5.0,
        device: str = "cuda",
    ) -> None:
        self.repo = Path(repo).resolve()
        self.reference_audio_path = str(Path(reference_audio).resolve())
        self.input_sample_rate = int(input_sample_rate)
        self.block_seconds = float(block_seconds)
        self.diffusion_steps = int(diffusion_steps)
        self.cfg_rate = float(cfg_rate)
        self.prompt_seconds = float(prompt_seconds)
        self.device = torch.device(device)
        self._lock = threading.Lock()

        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))

        # IMTalker globally wraps Wav2Vec2Model.from_pretrained to inject an
        # attention keyword that its custom subclass strips. Seed-VC loads the
        # stock XLS-R class, where that keyword reaches __init__ and fails.
        # IMTalker's model is already constructed before this converter, so
        # restore the original HF loader for Seed-VC and subsequent stock loads.
        try:
            from generator import wav2vec2 as imtalker_wav2vec
            from transformers import Wav2Vec2Model

            Wav2Vec2Model.from_pretrained = classmethod(imtalker_wav2vec.orig_func)
        except (ImportError, AttributeError):
            pass

        spec = importlib.util.spec_from_file_location(
            "seedvc_realtime_module", self.repo / "real-time-gui.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("Unable to load Seed-VC real-time module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.device = self.device
        self.module = module

        class Args:
            checkpoint_path = None
            config_path = None
            fp16 = True

        start = time.perf_counter()
        old_cwd = Path.cwd()
        try:
            import os

            os.chdir(self.repo)
            self.model_set = module.load_models(Args())
        finally:
            os.chdir(old_cwd)

        self.model_sr = int(self.model_set[-1]["sampling_rate"])
        self.zc = self.model_sr // 50
        self.block_frame = round(self.block_seconds * 50) * self.zc
        self.crossfade_frame = round(0.04 * 50) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = round(2.5 * 50) * self.zc
        self.extra_right_frame = round(0.02 * 50) * self.zc
        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_right_frame // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc

        total_model_samples = (
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
            + self.extra_right_frame
        )
        self.input_wav = torch.zeros(
            total_model_samples, device=self.device, dtype=torch.float32
        )
        self.input_wav_res = torch.zeros(
            round(total_model_samples * 16_000 / self.model_sr),
            device=self.device,
            dtype=torch.float32,
        )
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=self.device, dtype=torch.float32
        )
        fade = torch.linspace(
            0.0,
            1.0,
            steps=self.sola_buffer_frame,
            device=self.device,
            dtype=torch.float32,
        )
        self.fade_in = torch.sin(0.5 * torch.pi * fade).square()
        self.fade_out = 1.0 - self.fade_in
        self.reference_wav, _ = librosa.load(
            self.reference_audio_path, sr=self.model_sr, mono=True
        )
        self._first = True
        self.load_seconds = time.perf_counter() - start

    @torch.inference_mode()
    def reset(self) -> None:
        with self._lock:
            self.input_wav.zero_()
            self.input_wav_res.zero_()
            self.sola_buffer.zero_()
            self._first = True

    @torch.inference_mode()
    def convert(self, pcm: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        with self._lock:
            started = time.perf_counter()
            source = np.asarray(pcm, dtype=np.float32).reshape(-1)
            expected = round(self.block_seconds * self.input_sample_rate)
            if source.size < expected:
                source = np.pad(source, (0, expected - source.size))
            elif source.size > expected:
                source = source[:expected]

            source_t = torch.from_numpy(source).to(self.device)
            source_model = torchaudio.functional.resample(
                source_t, self.input_sample_rate, self.model_sr
            )
            if source_model.numel() < self.block_frame:
                source_model = F.pad(source_model, (0, self.block_frame - source_model.numel()))
            source_model = source_model[: self.block_frame]
            self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame :].clone()
            self.input_wav[-self.block_frame :] = source_model
            self.input_wav_res = torchaudio.functional.resample(
                self.input_wav, self.model_sr, 16_000
            )

            old_cwd = Path.cwd()
            try:
                import os

                os.chdir(self.repo)
                infer_wav = self.module.custom_infer(
                    self.model_set,
                    self.reference_wav,
                    self.reference_audio_path,
                    self.input_wav_res,
                    round(self.block_seconds * 16_000),
                    self.skip_head,
                    self.skip_tail,
                    self.return_length,
                    self.diffusion_steps,
                    self.cfg_rate,
                    self.prompt_seconds,
                    2.0,
                )
            finally:
                os.chdir(old_cwd)

            conv_input = infer_wav[None, None, : self.sola_buffer_frame + self.sola_search_frame]
            if self._first or self.sola_buffer_frame == 0:
                sola_offset = 0
            else:
                numerator = F.conv1d(conv_input, self.sola_buffer[None, None, :])
                denominator = torch.sqrt(
                    F.conv1d(
                        conv_input.square(),
                        torch.ones(
                            1,
                            1,
                            self.sola_buffer_frame,
                            device=self.device,
                        ),
                    )
                    + 1e-8
                )
                sola_offset = int(torch.argmax(numerator[0, 0] / denominator[0, 0]).item())

            infer_wav = infer_wav[sola_offset:]
            infer_wav[: self.sola_buffer_frame] *= self.fade_in
            infer_wav[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out
            self.sola_buffer.copy_(
                infer_wav[self.block_frame : self.block_frame + self.sola_buffer_frame]
            )
            self._first = False
            output_model = infer_wav[: self.block_frame]
            output_24k = torchaudio.functional.resample(
                output_model, self.model_sr, self.input_sample_rate
            )
            output = output_24k.detach().float().cpu().numpy()
            if output.size < expected:
                output = np.pad(output, (0, expected - output.size))
            output = np.clip(output[:expected], -1.0, 1.0).astype(np.float32)
            elapsed = time.perf_counter() - started
            return output, {
                "seedvc_ms": elapsed * 1000.0,
                "seedvc_rtf": elapsed / self.block_seconds,
                "sola_offset": float(sola_offset),
            }
