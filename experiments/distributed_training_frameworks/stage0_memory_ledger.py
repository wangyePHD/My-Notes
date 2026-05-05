import argparse

GI_B = 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 0 demo: estimate training memory for model states and activations."
    )
    parser.add_argument(
        "--num-params-b",
        type=float,
        default=7.0,
        help="Model size in billions of parameters.",
    )
    parser.add_argument(
        "--weight-bytes",
        type=int,
        default=2,
        help="Bytes per weight element. BF16/FP16 usually use 2.",
    )
    parser.add_argument(
        "--grad-bytes",
        type=int,
        default=2,
        help="Bytes per gradient element.",
    )
    parser.add_argument(
        "--optimizer-bytes-per-param",
        type=int,
        default=8,
        help="Bytes per parameter used by optimizer states. FP32 Adam moments are typically 8.",
    )
    parser.add_argument(
        "--master-weight-bytes",
        type=int,
        default=0,
        help="Extra bytes per parameter for FP32 master weights. FP16 training often uses 4.",
    )
    parser.add_argument(
        "--activation-gib",
        type=float,
        default=10.0,
        help="Rough per-rank activation memory estimate in GiB.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=8,
        help="How many ranks you plan to train with.",
    )
    return parser.parse_args()


def to_gib(num_bytes: float) -> float:
    return num_bytes / GI_B


def format_gib(value: float) -> str:
    return f"{value:,.2f} GiB"


def main() -> None:
    args = parse_args()
    num_params = args.num_params_b * 1_000_000_000

    params_gib = to_gib(num_params * args.weight_bytes)
    grads_gib = to_gib(num_params * args.grad_bytes)
    optimizer_gib = to_gib(num_params * args.optimizer_bytes_per_param)
    master_gib = to_gib(num_params * args.master_weight_bytes)

    model_states_gib = params_gib + grads_gib + optimizer_gib + master_gib
    ddp_per_rank_gib = model_states_gib + args.activation_gib
    ideal_full_shard_lower_bound_gib = model_states_gib / args.num_gpus + args.activation_gib

    print("Stage 0 memory ledger")
    print("=" * 80)
    print(f"Model size                 : {args.num_params_b:.2f}B parameters")
    print(f"Weight bytes / param       : {args.weight_bytes}")
    print(f"Gradient bytes / param     : {args.grad_bytes}")
    print(f"Optimizer bytes / param    : {args.optimizer_bytes_per_param}")
    print(f"Master weight bytes / param: {args.master_weight_bytes}")
    print(f"Activation estimate        : {args.activation_gib:.2f} GiB per rank")
    print(f"GPU count                  : {args.num_gpus}")
    print("-" * 80)
    print(f"Parameters                 : {format_gib(params_gib)}")
    print(f"Gradients                  : {format_gib(grads_gib)}")
    print(f"Optimizer states           : {format_gib(optimizer_gib)}")
    print(f"Master weights             : {format_gib(master_gib)}")
    print(f"Model states total         : {format_gib(model_states_gib)}")
    print("-" * 80)
    print(f"DDP per-rank baseline      : {format_gib(ddp_per_rank_gib)}")
    print(
        "Ideal full-shard lower bound: "
        f"{format_gib(ideal_full_shard_lower_bound_gib)}"
    )
    print("-" * 80)
    print("Notes:")
    print("1. DDP keeps full model states on every rank, so memory does not shrink with GPU count.")
    print("2. The full-shard number above is a lower bound, not a real peak memory guarantee.")
    print("3. Real training also pays for temp buffers, communication buffers, and allocator fragmentation.")


if __name__ == "__main__":
    main()
