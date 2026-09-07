from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator.unitalk_wav2vec_adapter import UniTalkWav2VecAdapter


def setup_ddp() -> tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        return 0, 1, 0, device
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank, torch.device(f"cuda:{local_rank}")


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tensor(path: Path, *keys: str) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload
    for key in keys:
        value = payload.get(key)
        if isinstance(value, torch.Tensor):
            return value
    raise RuntimeError(f"No tensor key {keys} in {path}")


class UnitalkAllLayerDataset(Dataset):
    def __init__(self, dataset_root: str, limit: int = 0, rms_dir: str = "") -> None:
        self.root = Path(dataset_root)
        self.helium_dir = self.root / "helium"
        self.teacher_dir = self.root / "w2v_all_layers_50hz"
        self.rms_dir = Path(rms_dir) if rms_dir else None
        stems = []
        for path in sorted(self.helium_dir.glob("*_helium.pt")):
            stem = path.name[: -len("_helium.pt")]
            if (self.teacher_dir / f"{stem}_w2v_all_layers.pt").exists():
                stems.append(stem)
        if limit > 0:
            stems = stems[:limit]
        if not stems:
            raise RuntimeError(f"No matched helium/all-layer teacher files under {self.root}")
        self.stems = stems

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        helium = load_tensor(self.helium_dir / f"{stem}_helium.pt", "helium_states", "helium").float()
        teacher = load_tensor(self.teacher_dir / f"{stem}_w2v_all_layers.pt", "w2v_all_layers").float()
        item = {
            "stem": stem,
            "helium": helium,
            "teacher": teacher,
            "target_len": int(teacher.shape[0]),
        }
        if self.rms_dir is not None:
            item["silence"] = load_tensor(
                self.rms_dir / f"{stem}_rms.pt", "silence"
            ).bool()
        return item


def collate(batch: list[dict]) -> dict:
    bsz = len(batch)
    h_len = max(int(x["helium"].shape[0]) for x in batch)
    t_len = max(int(x["teacher"].shape[0]) for x in batch)
    helium = torch.zeros(bsz, h_len, 4096, dtype=torch.float32)
    teacher = torch.zeros(bsz, t_len, 12, 768, dtype=torch.float32)
    mask = torch.zeros(bsz, t_len, dtype=torch.bool)
    silence = (
        torch.zeros(bsz, t_len, dtype=torch.bool)
        if "silence" in batch[0]
        else None
    )
    stems = []
    for i, item in enumerate(batch):
        h = item["helium"]
        y = item["teacher"]
        helium[i, : h.shape[0]] = h
        teacher[i, : y.shape[0]] = y
        mask[i, : y.shape[0]] = True
        if silence is not None:
            cur_silence = item["silence"]
            silence[i, : cur_silence.shape[0]] = cur_silence
        stems.append(item["stem"])
    result = {
        "stems": stems,
        "helium": helium,
        "teacher": teacher,
        "mask": mask,
    }
    if silence is not None:
        result["silence"] = silence
    return result


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < pred.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(pred.dtype)
    valid = mask.sum() * math.prod(pred.shape[2:])
    return ((pred - target).square() * mask).sum() / valid.clamp_min(1.0)


def masked_cos(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    cos = F.cosine_similarity(pred.float(), target.float(), dim=-1)
    while mask.ndim < cos.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.to(cos.dtype)
    valid = mask.sum() * cos.shape[-1]
    return 1.0 - (cos * mask).sum() / valid.clamp_min(1.0)


def save_ckpt(model, optimizer, epoch: int, save_dir: Path, args, rank: int) -> None:
    if rank != 0:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    payload = {
        "adapter": raw.state_dict(),
        "model": raw.state_dict(),
        "epoch": epoch,
        "args": vars(args),
    }
    if args.save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    path = save_dir / ("last.pt" if args.last_only else f"epoch_{epoch:06d}.pt")
    torch.save(payload, path)
    if not args.last_only:
        torch.save(
            {
                "adapter": raw.state_dict(),
                "model": raw.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            },
            save_dir / "last.pt",
        )
    print(f"[save] {path}", flush=True)


def reduce_float(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= dist.get_world_size()
    return float(t.item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lambda_mse", type=float, default=1.0)
    parser.add_argument("--lambda_cos", type=float, default=0.1)
    parser.add_argument("--checkpoint_epochs", default="5,10")
    parser.add_argument(
        "--center_loss_frames",
        type=int,
        default=0,
        help="Compute loss only on this many centered teacher frames; 0 uses all frames.",
    )
    parser.add_argument(
        "--tail_loss_frames",
        type=int,
        default=0,
        help="Compute loss on this many frames near the sequence tail.",
    )
    parser.add_argument(
        "--future_context_frames",
        type=int,
        default=0,
        help="Frames after the tail-loss region that are context only.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--rms_dir", default="")
    parser.add_argument(
        "--silence_weight",
        type=float,
        default=0.0,
        help="Additional loss weight assigned to RMS-labeled silence frames.",
    )
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--right_context_frames",
        type=int,
        default=-1,
        help="-1 is bidirectional; 0 is causal; positive values permit that many future 50 Hz frames.",
    )
    parser.add_argument("--recurrent_layers", type=int, default=0)
    parser.add_argument("--lambda_velocity", type=float, default=0.0)
    parser.add_argument("--lambda_chunk_consistency", type=float, default=0.0)
    parser.add_argument("--consistency_crop_frames", type=int, default=48)
    parser.add_argument("--save_optimizer", action="store_true")
    parser.add_argument("--last_only", action="store_true")
    args = parser.parse_args()

    rank, world, local_rank, device = setup_ddp()
    seed_all(args.seed + rank)

    dataset = UnitalkAllLayerDataset(
        args.dataset_root, limit=args.limit, rms_dir=args.rms_dir
    )
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) if world > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate,
    )

    model = UniTalkWav2VecAdapter(
        dropout=0.0,
        right_context_frames=args.right_context_frames,
        recurrent_layers=args.recurrent_layers,
    ).to(device)
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu")
        state = payload.get(
            "adapter", payload.get("model", payload)
        ) if isinstance(payload, dict) else payload
        normalized = {
            key.removeprefix("module."): value for key, value in state.items()
        }
        incompatible = model.load_state_dict(
            normalized, strict=args.recurrent_layers == 0
        )
        if rank == 0:
            print(f"[resume] adapter={args.resume}", flush=True)
            if args.recurrent_layers > 0:
                print(f"[resume] non-strict={incompatible}", flush=True)
    if world > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda", enabled=args.precision == "fp16")
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    use_amp = args.precision != "fp32"
    checkpoint_epochs = {int(x) for x in args.checkpoint_epochs.split(",") if x.strip()}

    if rank == 0:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
        print(f"[start] samples={len(dataset)} world={world} batch_per_rank={args.batch_size}", flush=True)

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        loss_sum = mse_sum = cos_sum = 0.0
        count = 0
        for step, batch in enumerate(loader, start=1):
            helium = batch["helium"].to(device, non_blocking=True)
            teacher = batch["teacher"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            if args.center_loss_frames > 0 and args.tail_loss_frames > 0:
                raise ValueError("center_loss_frames and tail_loss_frames are mutually exclusive")
            if args.center_loss_frames > 0:
                center_mask = torch.zeros_like(mask)
                for batch_index in range(mask.shape[0]):
                    valid_len = int(mask[batch_index].sum().item())
                    width = min(int(args.center_loss_frames), valid_len)
                    start = max(0, (valid_len - width) // 2)
                    center_mask[batch_index, start : start + width] = True
                mask = mask & center_mask
            elif args.tail_loss_frames > 0:
                tail_mask = torch.zeros_like(mask)
                for batch_index in range(mask.shape[0]):
                    valid_len = int(mask[batch_index].sum().item())
                    end = max(0, valid_len - int(args.future_context_frames))
                    width = min(int(args.tail_loss_frames), end)
                    start = max(0, end - width)
                    tail_mask[batch_index, start:end] = True
                mask = mask & tail_mask
            loss_mask = mask.float()
            silence = batch.get("silence")
            if silence is not None and args.silence_weight > 0:
                silence = silence.to(device, non_blocking=True)
                loss_mask = loss_mask * (
                    1.0 + float(args.silence_weight) * silence.float()
                )
            target_len = int(teacher.shape[1])
            helium = F.interpolate(
                helium.transpose(1, 2),
                size=target_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2).contiguous()

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                pred = model(helium)
                mse = masked_mse(pred.float(), teacher.float(), loss_mask)
                cos = masked_cos(pred.float(), teacher.float(), loss_mask)
                loss = args.lambda_mse * mse + args.lambda_cos * cos
                if args.lambda_velocity > 0:
                    velocity_mask = loss_mask[:, 1:] * loss_mask[:, :-1]
                    velocity = masked_mse(
                        pred[:, 1:] - pred[:, :-1],
                        teacher[:, 1:] - teacher[:, :-1],
                        velocity_mask,
                    )
                    loss = loss + args.lambda_velocity * velocity
                else:
                    velocity = pred.new_zeros(())
                if args.lambda_chunk_consistency > 0:
                    crop = min(
                        int(args.consistency_crop_frames),
                        max(0, helium.shape[1] - 2),
                    )
                    if crop > 0:
                        cropped_pred = model(helium[:, crop:])
                        consistency_mask = loss_mask[:, crop:]
                        consistency = masked_mse(
                            cropped_pred,
                            pred[:, crop:].detach(),
                            consistency_mask,
                        )
                        loss = loss + args.lambda_chunk_consistency * consistency
                    else:
                        consistency = pred.new_zeros(())
                else:
                    consistency = pred.new_zeros(())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach().item())
            mse_sum += float(mse.detach().item())
            cos_sum += float(cos.detach().item())
            count += 1
            if rank == 0 and (step == 1 or step % 25 == 0):
                print(
                    f"[epoch {epoch:03d}] step {step:04d}/{len(loader)} "
                    f"loss={loss_sum/count:.5f} mse={mse_sum/count:.5f} "
                    f"cos={cos_sum/count:.5f} vel={float(velocity.detach()):.5f} "
                    f"cons={float(consistency.detach()):.5f}",
                    flush=True,
                )

        loss_avg = reduce_float(loss_sum / max(1, count), device)
        mse_avg = reduce_float(mse_sum / max(1, count), device)
        cos_avg = reduce_float(cos_sum / max(1, count), device)
        if rank == 0:
            print(f"[epoch {epoch:03d} done] loss={loss_avg:.6f} mse={mse_avg:.6f} cos={cos_avg:.6f} time={time.time()-t0:.1f}s", flush=True)
        if epoch in checkpoint_epochs or epoch == args.epochs:
            save_ckpt(model, optimizer, epoch, Path(args.save_dir) / "checkpoints", args, rank)

    cleanup_ddp()


if __name__ == "__main__":
    main()
