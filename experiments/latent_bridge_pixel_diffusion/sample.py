import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from latent_bridge import FlowSchedule, LatentBridgeDiT
from latent_bridge.utils import load_checkpoint, save_image_grid, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="samples/lbpd")
    parser.add_argument("--num", type=int, default=16)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--amp", choices=["none", "fp16", "bf16"], default="bf16")
    return parser.parse_args()


def amp_dtype(mode: str):
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    return None


def velocity_from_xpred(x_t: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    while t.ndim < x_t.ndim:
        t = t[..., None]
    return (x0 - x_t) / (1.0 - t).clamp(min=1e-3)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    ckpt = load_checkpoint(args.ckpt, device)
    train_args = ckpt["args"]

    model = LatentBridgeDiT(
        image_size=train_args["image_size"],
        bridge_size=train_args["bridge_size"],
        patch_size=train_args["patch_size"],
        dim=train_args["dim"],
        depth=train_args["depth"],
        heads=train_args["heads"],
        mlp_ratio=train_args["mlp_ratio"],
        dropout=train_args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    schedule = FlowSchedule(cascade_ratio=train_args["cascade_ratio"])
    dtype = amp_dtype(args.amp)

    z_bridge = torch.randn(args.num, 3, train_args["bridge_size"], train_args["bridge_size"], device=device)
    z_pixel = torch.randn(args.num, 3, train_args["image_size"], train_args["image_size"], device=device)

    for step in tqdm(range(args.steps), desc="sampling"):
        _, t_bridge, t_pixel = schedule.step_times(step, args.steps, device)
        dt_bridge, dt_pixel = schedule.step_delta(step, args.steps, device)
        t_bridge_b = t_bridge.expand(args.num)
        t_pixel_b = t_pixel.expand(args.num)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            pred_pixel, pred_bridge = model(z_pixel, z_bridge, t_pixel_b, t_bridge_b)

        if dt_bridge.item() > 0:
            v_bridge = velocity_from_xpred(z_bridge, pred_bridge.float(), t_bridge_b)
            z_bridge = z_bridge + dt_bridge * v_bridge
        if dt_pixel.item() > 0:
            v_pixel = velocity_from_xpred(z_pixel, pred_pixel.float(), t_pixel_b)
            z_pixel = z_pixel + dt_pixel * v_pixel

    if train_args.get("predict_residual", False):
        up_bridge = F.interpolate(z_bridge, size=(train_args["image_size"], train_args["image_size"]), mode="bicubic", align_corners=False)
        image = up_bridge + z_pixel
    else:
        image = z_pixel

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_image_grid(image, out / "grid.png")
    for i, sample in enumerate(image):
        save_image_grid(sample.unsqueeze(0), out / f"{i:04d}.png", nrow=1)


if __name__ == "__main__":
    main()
