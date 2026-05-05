import json
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CachedFeatureLatentDataset(Dataset):
    """Image dataset paired with deterministic frozen-encoder latents.

    The latent is not a low-resolution RGB target and not a decoder bottleneck.
    It is a representation target, e.g. DINO/SigLIP/MAE patch tokens cached by
    cache_feature_latents.py.
    """

    def __init__(
        self,
        image_root: str,
        latent_cache: str,
        image_size: int,
        random_flip: bool = False,
    ) -> None:
        self.image_root = Path(image_root)
        self.latent_cache = Path(latent_cache)
        self.items = self._read_metadata(self.latent_cache / "metadata.jsonl")
        if not self.items:
            raise FileNotFoundError(f"No cached latent metadata found in {self.latent_cache}")

        aug = [transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC)]
        aug.append(transforms.CenterCrop(image_size))
        if random_flip:
            aug.append(transforms.RandomHorizontalFlip())
        aug.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.transform = transforms.Compose(aug)

    @staticmethod
    def _read_metadata(path: Path) -> List[Dict[str, str]]:
        if not path.exists():
            return []
        rows: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.items)

    def latent_shape(self) -> tuple[int, int]:
        sample = torch.load(self.latent_cache / self.items[0]["latent"], map_location="cpu")
        latent = sample["latent"] if isinstance(sample, dict) else sample
        return int(latent.shape[0]), int(latent.shape[1])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        image_path = self.image_root / item["image"]
        latent_path = self.latent_cache / item["latent"]

        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))

        payload = torch.load(latent_path, map_location="cpu")
        latent = payload["latent"] if isinstance(payload, dict) else payload
        return {
            "image": image,
            "latent": latent.float(),
        }
