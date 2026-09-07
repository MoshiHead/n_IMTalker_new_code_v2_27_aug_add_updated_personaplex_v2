from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


def load_tensor(path: Path, *keys: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload
    for key in keys:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise RuntimeError(f"No tensor key {keys} in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--threshold_db", type=float, default=-50.0)
    parser.add_argument("--frame_seconds", type=float, default=0.02)
    args = parser.parse_args()

    root = Path(args.dataset_root)
    wav_dir = root / "reply_wav_24k"
    teacher_dir = root / "w2v_all_layers_50hz"
    out_dir = root / "rms_50hz"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(wav_dir.glob("*.wav"))
    made = 0
    for index, wav_path in enumerate(wav_files, start=1):
        stem = wav_path.stem
        teacher_path = teacher_dir / f"{stem}_w2v_all_layers.pt"
        if not teacher_path.exists():
            continue
        out_path = out_dir / f"{stem}_rms.pt"
        if out_path.exists():
            continue

        target_len = int(
            load_tensor(teacher_path, "w2v_all_layers").shape[0]
        )
        wav, sample_rate = torchaudio.load(str(wav_path))
        wav = wav.mean(dim=0)
        frame_samples = max(1, int(round(sample_rate * args.frame_seconds)))
        frame_count = int(wav.shape[0]) // frame_samples
        wav = wav[: frame_count * frame_samples]
        rms = wav.view(frame_count, frame_samples).square().mean(dim=1).sqrt()
        rms_db = 20.0 * torch.log10(rms + 1e-8)
        rms_db = F.interpolate(
            rms_db.view(1, 1, -1),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).view(-1)
        silence = rms_db < float(args.threshold_db)
        torch.save(
            {
                "rms_db": rms_db.to(torch.float16),
                "silence": silence.to(torch.uint8),
                "threshold_db": float(args.threshold_db),
                "source_wav": str(wav_path),
            },
            out_path,
        )
        made += 1
        if index == 1 or index % 250 == 0:
            print(f"[rms] {index}/{len(wav_files)} made={made}", flush=True)

    print(f"[done] files={len(wav_files)} made={made}", flush=True)


if __name__ == "__main__":
    main()
