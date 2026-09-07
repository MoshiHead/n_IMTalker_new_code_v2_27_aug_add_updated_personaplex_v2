from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generator.options.base_options import BaseOptions
from generator.FM import FMGenerator
from renderer.models import IMTRenderer
from liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary import StudioNativeLiveAdapter


def make_opt(args):
    parser = argparse.ArgumentParser()
    parser = BaseOptions().initialize(parser)
    opt, _ = parser.parse_known_args([])
    opt.rank = 0
    opt.ngpus = 1
    opt.generator_path = args.generator_path
    opt.renderer_path = args.renderer_path
    opt.wav2vec_model_path = args.wav2vec_ckpt
    opt.only_last_features = True
    opt.audio_adapter_mode = "none"
    opt.audio_feat_dim = 768
    opt.fix_noise_seed = bool(args.fix_noise_seed)
    opt.seed = int(args.seed)
    opt.nfe = int(args.nfe)
    opt.a_cfg_scale = float(args.a_cfg_scale)
    opt.wav2vec_sec = float(args.wav2vec_sec)
    return opt


def load_renderer(renderer, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("gen.", ""): v for k, v in state.items() if k.startswith("gen.")}
    missing, unexpected = renderer.load_state_dict(state, strict=False)
    print(f"[load] renderer missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def load_generator(generator, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    stripped = {}
    for k, v in state.items():
        if k.startswith("model."):
            stripped[k[len("model."):]] = v
        else:
            stripped[k] = v
    cur = generator.state_dict()
    loadable = {k: v for k, v in stripped.items() if k in cur and tuple(cur[k].shape) == tuple(v.shape)}
    missing, unexpected = generator.load_state_dict(loadable, strict=False)
    print(f"[load] generator loaded={len(loadable)} missing={len(missing)} unexpected={len(unexpected)}", flush=True)


def load_image_tensor(ref_path, size=512):
    img = cv2.imread(str(ref_path))
    if img is None:
        raise RuntimeError(f"could not read {ref_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img).convert("RGB")
    tfm = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])
    return tfm(pil).unsqueeze(0)


@torch.no_grad()
def encode_image(renderer, image):
    dense, identity = renderer.dense_feature_encoder(image)
    ref_x = renderer.latent_token_encoder(image)
    return dense, ref_x, identity


@torch.no_grad()
def decode_motion(renderer, dense, ref_x, motion, identity, sub_batch=8):
    ref_adapted = renderer.adapt(ref_x, identity)
    ref_decoded = renderer.latent_token_decoder(ref_adapted)
    frames = []
    # Renderer API is per frame, but keep periodic prints for long sessions.
    for t in range(motion.shape[1]):
        cur = renderer.adapt(motion[:, t], identity)
        cur_decoded = renderer.latent_token_decoder(cur)
        frames.append(renderer.decode(cur_decoded, ref_decoded, dense))
        if (t + 1) % 100 == 0:
            print(f"[render] {t+1}/{motion.shape[1]}", flush=True)
    return torch.stack(frames, dim=1).squeeze(0)


def save_video(frames, video_path, fps, audio_path=None):
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = Path(tmp.name)
    vid = frames.detach().clamp(-1, 1).cpu().permute(0, 2, 3, 1)
    vid = (vid * 255).to(torch.uint8)
    torchvision.io.write_video(str(temp_path), vid, fps=float(fps))
    if audio_path and Path(audio_path).exists():
        subprocess.run([
            "ffmpeg", "-y", "-i", str(temp_path), "-i", str(audio_path),
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(video_path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_path.unlink(missing_ok=True)
    else:
        os.replace(temp_path, video_path)


def load_adapter(args, device):
    adapter = StudioNativeLiveAdapter(args.wav2vec_ckpt, args.adapter_num_layers, args.adapter_dropout).to(device).eval()
    payload = torch.load(args.adapter_path, map_location="cpu")
    if isinstance(payload, dict):
        state = payload.get("adapter", payload.get("model", payload))
    else:
        state = payload
    adapter.load_state_dict(state, strict=True)
    print(f"[load] adapter={args.adapter_path}", flush=True)
    return adapter


@torch.no_grad()
def motion_batch(generator, adapter, helium, ref_x, target_frames, args):
    print(f"[batch] helium={tuple(helium.shape)} target_frames={target_frames}", flush=True)
    _frontend50, _final50, final25 = adapter.forward_single(helium, target_frames)
    motion = generator.sample_from_wav2vec_features(
        {"ref_x": ref_x, "pose": None, "cam": None, "gaze": None},
        final25.unsqueeze(0).float(),
        a_cfg_scale=float(args.a_cfg_scale),
        nfe=int(args.nfe),
        seed=int(args.seed),
    )[:, :target_frames]
    return motion, {"adapter_final25": final25.cpu()}


@torch.no_grad()
def motion_chunk(generator, adapter, helium, ref_x, chunks, args):
    deque_size = int(args.helium_deque_size)
    helium_deque = torch.zeros(deque_size, helium.shape[1], device=helium.device, dtype=torch.float32)
    helium_deque_filled = 0
    stream_state = None
    abs_frame = 0
    ptr = 0
    motions = []
    adapter_parts = []
    for i, row in enumerate(chunks):
        target_frames = int(row.get("frames", args.fm_chunk_frames))
        # Live uses 12.5Hz Helium and 25fps video, so normally 12 tokens -> 24 frames.
        steps = max(1, int(round(target_frames * 12.5 / 25.0)))
        cur = helium[ptr:min(ptr + steps, helium.shape[0])]
        ptr += int(cur.shape[0])
        if cur.numel() == 0:
            break
        current_steps = int(cur.shape[0])
        if current_steps >= deque_size:
            helium_deque = cur[-deque_size:].detach().clone()
            helium_deque_filled = deque_size
        else:
            helium_deque = torch.cat([helium_deque[current_steps:], cur], dim=0).contiguous()
            helium_deque_filled = min(deque_size, int(helium_deque_filled) + current_steps)
        target_len_full = deque_size * 2
        _frontend50, _final50, full25 = adapter.forward_single(helium_deque, target_len_full)
        fresh_frames = max(1, current_steps * 2)
        feat25 = full25[-fresh_frames:].contiguous()
        if int(feat25.shape[0]) != target_frames:
            feat25 = F.interpolate(feat25.T.unsqueeze(0), size=target_frames, mode="linear", align_corners=False).squeeze(0).T.contiguous()
        data = {"a_feat": feat25.unsqueeze(0).float(), "ref_x": ref_x}
        motion, stream_state = generator.sample(
            data,
            a_cfg_scale=float(args.a_cfg_scale),
            nfe=int(args.nfe),
            seed=int(args.seed),
            stream_state=stream_state,
            return_state=True,
        )
        motion = motion[:, :target_frames].detach()
        motions.append(motion)
        adapter_parts.append(feat25.cpu())
        abs_frame += int(motion.shape[1])
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[chunk] {i+1}/{len(chunks)} ptr={ptr}/{helium.shape[0]} frames={abs_frame} deque={helium_deque_filled}", flush=True)
    if not motions:
        raise RuntimeError("chunk mode produced no motion")
    return torch.cat(motions, dim=1), {"adapter_final25_chunks": torch.cat(adapter_parts, dim=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--mode", choices=["batch", "chunk"], required=True)
    ap.add_argument("--generator_path", required=True)
    ap.add_argument("--renderer_path", required=True)
    ap.add_argument("--adapter_path", required=True)
    ap.add_argument("--wav2vec_ckpt", default="/workspace/IMTalker/checkpoints/wav2vec2-base-960h")
    ap.add_argument("--ref_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--adapter_num_layers", type=int, default=6)
    ap.add_argument("--adapter_dropout", type=float, default=0.1)
    ap.add_argument("--nfe", type=int, default=5)
    ap.add_argument("--a_cfg_scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--fix_noise_seed", action="store_true")
    ap.add_argument("--wav2vec_sec", type=float, default=0.96)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--fm_chunk_frames", type=int, default=24)
    ap.add_argument("--helium_deque_size", type=int, default=100)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dump = Path(args.dump_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    helium_pack = torch.load(dump / "full_helium_raw.pt", map_location="cpu")
    helium = helium_pack["helium"].float().to(device).contiguous()
    chunks = helium_pack.get("chunks") or json.loads((dump / "meta.json").read_text()).get("chunks", [])
    meta = json.loads((dump / "meta.json").read_text()) if (dump / "meta.json").exists() else {}
    target_frames = int(meta.get("motion_frames", int(round(helium.shape[0] * 2))))
    audio_path = dump / "full_moshi_reply_24k.wav"

    opt = make_opt(args)
    renderer = IMTRenderer(opt).to(device).eval()
    generator = FMGenerator(opt).to(device).eval()
    load_renderer(renderer, args.renderer_path)
    load_generator(generator, args.generator_path, device)
    adapter = load_adapter(args, device)

    image = load_image_tensor(args.ref_path).to(device)
    dense, ref_x, identity = encode_image(renderer, image)

    t0 = time.time()
    if args.mode == "batch":
        motion, extra = motion_batch(generator, adapter, helium, ref_x, target_frames, args)
    else:
        motion, extra = motion_chunk(generator, adapter, helium, ref_x, chunks, args)
        motion = motion[:, :target_frames]
    print(f"[motion] mode={args.mode} shape={tuple(motion.shape)} sec={time.time()-t0:.2f}", flush=True)

    frames = decode_motion(renderer, dense, ref_x, motion, identity)
    out_mp4 = out_dir / f"{args.mode}_offline_replay.mp4"
    save_video(frames, out_mp4, args.fps, audio_path)
    torch.save({"motion": motion.cpu(), "helium_path": str(dump / "full_helium_raw.pt"), "mode": args.mode, **extra}, out_dir / f"{args.mode}_offline_replay.pt")
    print(f"[done] {out_mp4}", flush=True)


if __name__ == "__main__":
    main()
