from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageFolderDataset(Dataset):
    """Loads images and returns high-resolution pixels plus a deterministic low-res bridge."""

    def __init__(
        self,
        root: str,
        image_size: int,
        bridge_size: int,
        random_flip: bool = True,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        self.bridge_size = bridge_size
        self.paths = self._collect_paths(self.root)
        if not self.paths:
            raise FileNotFoundError(f"No images found under {self.root}")

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
    def _collect_paths(root: Path) -> List[Path]:
        return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
        with Image.open(path) as img:
            x_hr = self.transform(img.convert("RGB"))

        x_bridge = F.interpolate(
            x_hr.unsqueeze(0),
            size=(self.bridge_size, self.bridge_size),
            mode="area",
        ).squeeze(0)

        return {
            "image": x_hr,
            "bridge": x_bridge,
        }
