import argparse
import os
from typing import Optional

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 0 demo: inspect collective semantics and process groups."
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "gloo", "nccl"],
        default="auto",
        help="Communication backend. Use gloo for CPU-only local demos.",
    )
    return parser.parse_args()


def pick_backend(name: str) -> str:
    if name != "auto":
        return name
    return "nccl" if torch.cuda.is_available() else "gloo"


def init_distributed(backend: str) -> tuple[int, int, int, torch.device]:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "This script must be launched with torchrun so RANK/WORLD_SIZE are set."
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL backend requires CUDA.")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend)
    return rank, world_size, local_rank, device


def ordered_print(rank: int, world_size: int, message: str) -> None:
    for turn in range(world_size):
        dist.barrier()
        if turn == rank:
            print(f"[rank {rank}] {message}", flush=True)
    dist.barrier()


def demo_broadcast(rank: int, world_size: int, device: torch.device) -> None:
    tensor = (
        torch.tensor([10.0, 20.0, 30.0], device=device)
        if rank == 0
        else torch.tensor([-1.0, -1.0, -1.0], device=device)
    )
    ordered_print(rank, world_size, f"broadcast before: {tensor.detach().cpu().tolist()}")
    dist.broadcast(tensor, src=0)
    ordered_print(rank, world_size, f"broadcast after : {tensor.detach().cpu().tolist()}")


def demo_all_reduce(rank: int, world_size: int, device: torch.device) -> None:
    tensor = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = float(world_size * (world_size + 1) // 2)
    ordered_print(
        rank,
        world_size,
        f"all_reduce sum: {tensor.item():.1f} (expected {expected:.1f})",
    )


def demo_all_gather(rank: int, world_size: int, device: torch.device) -> None:
    local = torch.tensor([float(rank), float(rank + 100)], device=device)
    gathered = [torch.zeros_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    full = torch.cat(gathered)
    ordered_print(
        rank,
        world_size,
        f"all_gather local={local.cpu().tolist()} full={full.cpu().tolist()}",
    )


def demo_reduce_scatter(rank: int, world_size: int, device: torch.device) -> None:
    chunk_size = 2
    input_tensor = torch.arange(world_size * chunk_size, dtype=torch.float32, device=device)
    input_tensor = input_tensor + rank * 100.0
    output = torch.zeros(chunk_size, dtype=torch.float32, device=device)

    try:
        dist.reduce_scatter_tensor(output, input_tensor, op=dist.ReduceOp.SUM)
        start = rank * chunk_size
        end = start + chunk_size
        ordered_print(
            rank,
            world_size,
            (
                "reduce_scatter "
                f"input={input_tensor.cpu().tolist()} "
                f"slice={start}:{end} "
                f"output={output.cpu().tolist()}"
            ),
        )
    except (RuntimeError, ValueError, NotImplementedError) as exc:
        if rank == 0:
            print(
                "[rank 0] reduce_scatter_tensor skipped: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        dist.barrier()


def build_even_odd_groups(
    world_size: int,
) -> tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup]]:
    even_ranks = list(range(0, world_size, 2))
    odd_ranks = list(range(1, world_size, 2))
    even_group = dist.new_group(ranks=even_ranks)
    odd_group = dist.new_group(ranks=odd_ranks) if odd_ranks else None
    return even_group, odd_group


def demo_subgroups(rank: int, world_size: int, device: torch.device) -> None:
    if world_size < 2:
        if rank == 0:
            print("[rank 0] subgroup demo skipped because world_size < 2", flush=True)
        dist.barrier()
        return

    even_group, odd_group = build_even_odd_groups(world_size)
    group = even_group if rank % 2 == 0 else odd_group
    group_name = "even" if rank % 2 == 0 else "odd"

    tensor = torch.tensor([float(rank)], device=device)
    dist.all_reduce(tensor, group=group, op=dist.ReduceOp.SUM)
    ordered_print(rank, world_size, f"subgroup {group_name} all_reduce -> {tensor.item():.1f}")


def cleanup() -> None:
    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    backend = pick_backend(args.backend)
    rank, world_size, local_rank, device = init_distributed(backend)

    if rank == 0:
        print(
            f"Launching Stage 0 demo with backend={backend}, world_size={world_size}",
            flush=True,
        )
    ordered_print(rank, world_size, f"device={device}, local_rank={local_rank}")

    demo_broadcast(rank, world_size, device)
    demo_all_reduce(rank, world_size, device)
    demo_all_gather(rank, world_size, device)
    demo_reduce_scatter(rank, world_size, device)
    demo_subgroups(rank, world_size, device)

    if rank == 0:
        print(
            "[rank 0] Demo complete. Focus on what each collective returns on every rank.",
            flush=True,
        )
    cleanup()


if __name__ == "__main__":
    main()
