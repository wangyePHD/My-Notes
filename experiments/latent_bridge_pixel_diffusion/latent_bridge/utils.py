import math
import os
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import torch
import torch.distributed as dist
from PIL import Image
from torchvision.utils import save_image


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_dist() else 0


def world_size() -> int:
    return dist.get_world_size() if is_dist() else 1


def is_main() -> bool:
    return rank() == 0


def setup_distributed() -> torch.device:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_distributed() -> None:
    if is_dist():
        dist.destroy_process_group()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, args: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": asdict(args) if is_dataclass(args) else vars(args),
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(-1, 1) + 1.0) * 0.5


def save_image_grid(x: torch.Tensor, path: Path, nrow: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if nrow is None:
        nrow = int(math.sqrt(x.shape[0]))
    save_image(denormalize(x), path, nrow=max(1, nrow))


def valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
