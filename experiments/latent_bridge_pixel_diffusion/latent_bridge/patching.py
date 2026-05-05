import torch


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"Image size {(h, w)} must be divisible by patch_size={patch_size}")
    ph = pw = patch_size
    x = x.reshape(b, c, h // ph, ph, w // pw, pw)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.reshape(b, (h // ph) * (w // pw), ph * pw * c)


def unpatchify(tokens: torch.Tensor, patch_size: int, height: int, width: int, channels: int = 3) -> torch.Tensor:
    b, n, d = tokens.shape
    gh, gw = height // patch_size, width // patch_size
    if n != gh * gw:
        raise ValueError(f"Expected {gh * gw} tokens, got {n}")
    expected = patch_size * patch_size * channels
    if d != expected:
        raise ValueError(f"Expected token dim {expected}, got {d}")
    x = tokens.reshape(b, gh, gw, patch_size, patch_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.reshape(b, channels, height, width)
