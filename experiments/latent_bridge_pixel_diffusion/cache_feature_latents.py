import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--encoder", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--encoder-size", type=int, default=518)
    parser.add_argument("--latent-pool", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def build_encoder(name: str, device: torch.device):
    import timm
    from timm.data import create_transform, resolve_data_config

    model = timm.create_model(name, pretrained=True, num_classes=0).to(device).eval()
    config = resolve_data_config({"input_size": (3, 518, 518)}, model=model)
    transform = create_transform(**config, is_training=False)
    return model, transform


def extract_tokens(model, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        feats = model.forward_features(x)
    if isinstance(feats, dict):
        for key in ("x_norm_patchtokens", "patch_tokens", "tokens"):
            if key in feats:
                feats = feats[key]
                break
        else:
            raise KeyError(f"Cannot find patch tokens in encoder output keys: {list(feats.keys())}")
    if isinstance(feats, (list, tuple)):
        feats = feats[-1]
    if feats.ndim == 4:
        b, c, h, w = feats.shape
        feats = feats.flatten(2).transpose(1, 2).contiguous()
    if feats.ndim != 3:
        raise ValueError(f"Expected latent tokens with shape [B, N, C], got {tuple(feats.shape)}")
    return feats


def pool_tokens(tokens: torch.Tensor, pool: int) -> torch.Tensor:
    if pool <= 1:
        return tokens
    b, n, c = tokens.shape
    side = int(n ** 0.5)
    if side * side != n:
        raise ValueError(f"Cannot spatially pool non-square token grid with N={n}")
    x = tokens.transpose(1, 2).reshape(b, c, side, side)
    x = F.avg_pool2d(x, kernel_size=pool, stride=pool)
    return x.flatten(2).transpose(1, 2).contiguous()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data)
    out = Path(args.out)
    latent_dir = out / "latents"
    latent_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    paths = collect_images(data_root)
    if not paths:
        raise FileNotFoundError(f"No images found under {data_root}")

    model, transform = build_encoder(args.encoder, device)
    metadata_path = out / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as meta:
        batch_images = []
        batch_rels = []
        batch_indices = []
        for idx, path in enumerate(tqdm(paths, desc="encoding")):
            with Image.open(path) as img:
                batch_images.append(transform(img.convert("RGB")))
            batch_rels.append(path.relative_to(data_root).as_posix())
            batch_indices.append(idx)
            if len(batch_images) < args.batch_size and idx != len(paths) - 1:
                continue

            x = torch.stack(batch_images).to(device)
            tokens = pool_tokens(extract_tokens(model, x).float().cpu(), args.latent_pool)
            for j, rel in enumerate(batch_rels):
                latent_name = f"latents/{batch_indices[j]:08d}.pt"
                torch.save(
                    {
                        "latent": tokens[j],
                        "image": rel,
                        "encoder": args.encoder,
                        "latent_pool": args.latent_pool,
                    },
                    out / latent_name,
                )
                meta.write(json.dumps({"image": rel, "latent": latent_name}, ensure_ascii=False) + "\n")

            batch_images.clear()
            batch_rels.clear()
            batch_indices.clear()

    print(f"Wrote {len(paths)} latents to {out}")


if __name__ == "__main__":
    main()
