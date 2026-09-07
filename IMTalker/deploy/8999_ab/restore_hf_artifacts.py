#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download


DATASET_REPO = "niloy629/hdtf_preprocess"


def download_dataset_file(filename: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=filename,
        )
    )
    if destination.exists() and destination.stat().st_size == cached.stat().st_size:
        return
    shutil.copy2(cached, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
    )
    args = parser.parse_args()

    checkpoint_root = args.project_root.resolve() / "checkpoints"
    live_root = checkpoint_root / "personaplex_imtalker_live_8999"

    download_dataset_file(
        "personaplex_imtalker_live_8999/checkpoints/"
        "unitalk_live_turn_silence_ft_epoch_000010.pt",
        live_root / "unitalk_live_turn_silence_ft_epoch_000010.pt",
    )
    download_dataset_file(
        "personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt",
        live_root / "silence_helium_mean.pt",
    )
    download_dataset_file(
        "lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt",
        checkpoint_root / "lora" / "ditto_blink_lora_withaudio_r64_1h_last.ckpt",
    )

    voice_root = checkpoint_root / "personaplex_voices"
    voice_path = voice_root / "NATM0.pt"
    if not voice_path.exists():
        archive = Path(
            hf_hub_download(
                repo_id="nvidia/personaplex-7b-v1",
                filename="voices.tgz",
            )
        )
        voice_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            member = next(
                item for item in tar.getmembers()
                if Path(item.name).name == "NATM0.pt"
            )
            member.name = "NATM0.pt"
            tar.extract(member, voice_root)

    required = [
        live_root / "unitalk_live_turn_silence_ft_epoch_000010.pt",
        live_root / "silence_helium_mean.pt",
        checkpoint_root / "lora" / "ditto_blink_lora_withaudio_r64_1h_last.ckpt",
        voice_path,
    ]
    for path in required:
        print(f"{path} {path.stat().st_size}")


if __name__ == "__main__":
    main()
