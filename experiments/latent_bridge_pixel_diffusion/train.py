import argparse
import math
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from latent_bridge import FlowSchedule, ImageFolderDataset, LatentBridgeDiT, add_noise, loss_weight
from latent_bridge.utils import (
    cleanup_distributed,
    is_dist,
    is_main,
    rank,
    save_checkpoint,
    seed_everything,
    setup_distributed,
    world_size,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--data", default="")
    parser.add_argument("--out", default="runs/lbpd")
    parser.add_argument("--resume", default="")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--bridge-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cascade-ratio", type=float, default=0.5)
    parser.add_argument("--predict-residual", action="store_true")
    parser.add_argument("--bridge-loss-weight", type=float, default=1.0)
    parser.add_argument("--pixel-loss-weight", type=float, default=1.0)
    parser.add_argument("--loss-weight", choices=["velocity", "none"], default="velocity")
    parser.add_argument("--unbalanced-times", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=5000)
    config_args, _ = parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r", encoding="utf-8") as f:
            defaults = yaml.safe_load(f) or {}
        parser.set_defaults(**{k.replace("-", "_"): v for k, v in defaults.items()})
    args = parser.parse_args()
    if not args.data:
        raise ValueError("--data is required, either on the CLI or in --config")
    return args


def amp_dtype(mode: str):
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    return None


def sample_global_times(batch: int, cascade_ratio: float, device: torch.device, balanced: bool) -> torch.Tensor:
    if not balanced:
        return torch.rand(batch, device=device)
    if batch == 1:
        if torch.rand((), device=device) < 0.5:
            return torch.rand(1, device=device) * cascade_ratio
        return cascade_ratio + torch.rand(1, device=device) * (1.0 - cascade_ratio)
    n_bridge = batch // 2
    n_pixel = batch - n_bridge
    first = torch.rand(n_bridge, device=device) * cascade_ratio
    second = cascade_ratio + torch.rand(n_pixel, device=device) * (1.0 - cascade_ratio)
    return torch.cat([first, second], dim=0)[torch.randperm(batch, device=device)]


def per_sample_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).pow(2).flatten(1).mean(dim=1)


def main() -> None:
    args = parse_args()
    device = setup_distributed()
    seed_everything(args.seed + rank())
    out_dir = Path(args.out)
    if is_main():
        out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageFolderDataset(args.data, args.image_size, args.bridge_size)
    sampler = DistributedSampler(dataset, num_replicas=world_size(), rank=rank(), shuffle=True) if is_dist() else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = LatentBridgeDiT(
        image_size=args.image_size,
        bridge_size=args.bridge_size,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))

    if is_dist():
        model = DDP(model, device_ids=[device.index], output_device=device.index)

    schedule = FlowSchedule(cascade_ratio=args.cascade_ratio)
    dtype = amp_dtype(args.amp)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp == "fp16"))

    step = start_step
    epoch = 0
    pbar = tqdm(total=args.steps, initial=step, disable=not is_main())
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1
        for batch in loader:
            if step >= args.steps:
                break
            image = batch["image"].to(device, non_blocking=True)
            bridge = batch["bridge"].to(device, non_blocking=True)
            if args.predict_residual:
                up_bridge = F.interpolate(bridge, size=(args.image_size, args.image_size), mode="bicubic", align_corners=False)
                pixel_target = image - up_bridge
            else:
                pixel_target = image

            global_t = sample_global_times(
                image.shape[0],
                args.cascade_ratio,
                device,
                balanced=not args.unbalanced_times,
            )
            t_bridge, t_pixel = schedule.times(global_t)
            bridge_t, _ = add_noise(bridge, t_bridge)
            pixel_t, _ = add_noise(pixel_target, t_pixel)

            active_bridge = (global_t < args.cascade_ratio).float()
            active_pixel = (global_t >= args.cascade_ratio).float()

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                pred_pixel, pred_bridge = model(pixel_t, bridge_t, t_pixel, t_bridge)
                pixel_loss = per_sample_mse(pred_pixel, pixel_target)
                bridge_loss = per_sample_mse(pred_bridge, bridge)
                pixel_w = loss_weight(t_pixel, args.loss_weight) * active_pixel
                bridge_w = loss_weight(t_bridge, args.loss_weight) * active_bridge
                pixel_loss = (pixel_loss * pixel_w).sum() / pixel_w.sum().clamp(min=1.0)
                bridge_loss = (bridge_loss * bridge_w).sum() / bridge_w.sum().clamp(min=1.0)
                loss = args.pixel_loss_weight * pixel_loss + args.bridge_loss_weight * bridge_loss
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if is_dist():
                metrics = torch.tensor([loss.item() * args.grad_accum, pixel_loss.item(), bridge_loss.item()], device=device)
                dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
                loss_value, pixel_value, bridge_value = metrics.tolist()
            else:
                loss_value, pixel_value, bridge_value = loss.item() * args.grad_accum, pixel_loss.item(), bridge_loss.item()

            step += 1
            if is_main() and step % args.log_every == 0:
                pbar.set_postfix(loss=f"{loss_value:.4f}", pixel=f"{pixel_value:.4f}", bridge=f"{bridge_value:.4f}")
            if is_main() and (step % args.save_every == 0 or step == args.steps):
                save_checkpoint(out_dir / "checkpoint_latest.pt", model, optimizer, step, args)
                save_checkpoint(out_dir / f"checkpoint_{step:08d}.pt", model, optimizer, step, args)
            pbar.update(1)

    pbar.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
