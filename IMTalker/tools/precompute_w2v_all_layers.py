from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchaudio

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator.wav2vec2 import Wav2VecModel


def load_tensor(path: Path, *keys: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload
    for key in keys:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise RuntimeError(f"No tensor key {keys} in {path}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--wav2vec_ckpt", default="/workspace/IMTalker/checkpoints/wav2vec2-base-960h")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "w2v_all_layers_50hz"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = root / "reply_wav_24k"
    final_dir = root / "w2v_final_50hz"
    wav_files = sorted(wav_dir.glob("*.wav"))
    wav_files = [p for i, p in enumerate(wav_files) if i % int(args.num_shards) == int(args.shard_idx)]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Wav2VecModel.from_pretrained(args.wav2vec_ckpt, local_files_only=True).to(device).eval().float()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    save_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    for index, wav_path in enumerate(wav_files, start=1):
        stem = wav_path.stem
        out_path = out_dir / f"{stem}_w2v_all_layers.pt"
        if out_path.exists():
            continue
        final_path = final_dir / f"{stem}_w2v_final.pt"
        target_len = int(load_tensor(final_path, "w2v_final").shape[0])

        wav, sr = torchaudio.load(str(wav_path))
        wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        wav = wav.to(device=device, dtype=torch.float32)

        outputs = model(wav, seq_len=target_len, output_hidden_states=True, return_dict=True)
        hidden = torch.stack([x[0] for x in outputs.hidden_states[-12:]], dim=1).detach().cpu().to(save_dtype)
        torch.save(
            {
                "w2v_all_layers": hidden,
                "source_wav": str(wav_path),
                "target_len": target_len,
                "layers": 12,
            },
            out_path,
        )
        if index == 1 or index % 100 == 0:
            print(f"[shard {args.shard_idx}/{args.num_shards}] {index}/{len(wav_files)} {stem} {tuple(hidden.shape)}", flush=True)


if __name__ == "__main__":
    main()
