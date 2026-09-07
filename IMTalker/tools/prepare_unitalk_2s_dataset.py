from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchaudio

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
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--wav2vec_ckpt",
        default="/workspace/IMTalker/checkpoints/wav2vec2-base-960h",
    )
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--chunk_seconds", type=float, default=2.0)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    helium_out = output_root / "helium"
    wav_out = output_root / "reply_wav_24k"
    teacher_out = output_root / "w2v_all_layers_50hz"
    for directory in (helium_out, wav_out, teacher_out):
        directory.mkdir(parents=True, exist_ok=True)

    helium_files = sorted((input_root / "helium").glob("*_helium.pt"))
    helium_files = [
        path
        for index, path in enumerate(helium_files)
        if index % args.num_shards == args.shard_idx
    ]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = (
        Wav2VecModel.from_pretrained(args.wav2vec_ckpt, local_files_only=True)
        .to(device)
        .eval()
        .float()
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    save_dtype = torch.float16 if args.save_dtype == "fp16" else torch.float32
    helium_rate = 12.5
    teacher_rate = 50
    helium_per_chunk = int(round(args.chunk_seconds * helium_rate))
    teacher_per_chunk = int(round(args.chunk_seconds * teacher_rate))

    made = skipped = 0
    for file_index, helium_path in enumerate(helium_files, start=1):
        stem = helium_path.name[: -len("_helium.pt")]
        wav_path = input_root / "reply_wav_24k" / f"{stem}.wav"
        if not wav_path.exists():
            skipped += 1
            continue

        helium = load_tensor(helium_path, "helium_states", "helium").float()
        wav, sample_rate = torchaudio.load(str(wav_path))
        wav = wav.mean(dim=0, keepdim=True)
        audio_per_chunk = int(round(args.chunk_seconds * sample_rate))
        chunk_count = min(
            int(helium.shape[0]) // helium_per_chunk,
            int(wav.shape[-1]) // audio_per_chunk,
        )

        for chunk_index in range(chunk_count):
            chunk_stem = f"{stem}__2s_{chunk_index:02d}"
            helium_path_out = helium_out / f"{chunk_stem}_helium.pt"
            wav_path_out = wav_out / f"{chunk_stem}.wav"
            teacher_path_out = teacher_out / f"{chunk_stem}_w2v_all_layers.pt"
            if (
                helium_path_out.exists()
                and wav_path_out.exists()
                and teacher_path_out.exists()
            ):
                continue

            h0 = chunk_index * helium_per_chunk
            a0 = chunk_index * audio_per_chunk
            helium_chunk = helium[h0 : h0 + helium_per_chunk].contiguous()
            wav_chunk = wav[:, a0 : a0 + audio_per_chunk].contiguous()

            wav_16k = wav_chunk
            if sample_rate != 16000:
                wav_16k = torchaudio.functional.resample(
                    wav_chunk, sample_rate, 16000
                )
            outputs = model(
                wav_16k.to(device=device, dtype=torch.float32),
                seq_len=teacher_per_chunk,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = (
                torch.stack([value[0] for value in outputs.hidden_states[-12:]], dim=1)
                .detach()
                .cpu()
                .to(save_dtype)
            )

            torch.save(
                {
                    "helium_states": helium_chunk.to(save_dtype),
                    "source_helium": str(helium_path),
                    "chunk_index": chunk_index,
                    "chunk_seconds": args.chunk_seconds,
                },
                helium_path_out,
            )
            torchaudio.save(str(wav_path_out), wav_chunk, sample_rate)
            torch.save(
                {
                    "w2v_all_layers": hidden,
                    "source_wav": str(wav_path),
                    "chunk_index": chunk_index,
                    "target_len": teacher_per_chunk,
                    "layers": 12,
                },
                teacher_path_out,
            )
            made += 1

        if file_index == 1 or file_index % 100 == 0:
            print(
                f"[shard {args.shard_idx}/{args.num_shards}] "
                f"{file_index}/{len(helium_files)} made={made} skipped={skipped}",
                flush=True,
            )

    print(
        f"[done shard {args.shard_idx}/{args.num_shards}] "
        f"files={len(helium_files)} made={made} skipped={skipped}",
        flush=True,
    )


if __name__ == "__main__":
    main()
