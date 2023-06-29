import torch
import torch.nn as nn
import argparse
import sys
import inspect
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--m",
    type=int,
    help="m dimension of A=m*k",
)
parser.add_argument(
    "--n",
    type=int,
    default="fp16",
    help="n dimension of B=k*n (out_features)",
)
parser.add_argument(
    "--k",
    type=int,
    help="k dimension of A=m*k and B=k*n (in_features), hidden_size",
)
parser.add_argument(
    "--group_size",
    type=int,
    help="Number of int4 values sharing the same scale and zero-point",
)

implem_group = parser.add_mutually_exclusive_group()
implem_group.add_argument(
    "--gptqforllama-path",
    type=str,
    default=None,
    help="Path to gptqforllama repo",
)
implem_group.add_argument(
    "--exllama-path",
    type=str,
    default=None,
    help="Path to exllama repo",
)
implem_group.add_argument(
    "--autogptq-path",
    type=str,
    default=None,
    help="Path to autogptq repo",
)
implem_group.add_argument(
    "--baseline",
    action="store_true",
    help="Use PyTorch fp16 x fp16",
)

parser.add_argument(
    "--act-order",
    type=str,
    choices=["yes", "no"],
    default="yes",
    help="Use act-order option (that reorders the groups, what for?)",
)
parser.add_argument(
    "--autogptq-implem",
    type=str,
    choices=["triton", "cuda-old", "cuda"],
    default=None,
    help="AutoGPTQ has three implementations. Choose which one to run.",
)
parser.add_argument(
    "--csv",
    action="store_true",
    help="Write results to a csv file",
)

bits = 4

args = parser.parse_args()

if args.act_order == "yes":
    act_order = True
else:
    act_order = False

if not act_order and args.gptqforllama_path:
    raise NotImplementedError("gptqforllama does not support no act-order")

if args.autogptq_path and not args.autogptq_implem:
    raise ValueError("Please pass --autogptq-implem {triton, cuda-old, cuda} when running autogptq implementation.")

# Please pardon me for this ugliness :)
if args.baseline:
    implementation = "baseline"
elif args.gptqforllama_path:
    implementation = "gptqforllama"
    sys.path.insert(0, args.gptqforllama_path)
    from quant.quant_linear import QuantLinear as GPTQForLlamaQuantLinear
    from quant.quant_linear import autotune_warmup_linear
elif args.autogptq_path:
    implementation = f"autogptq-{args.autogptq_implem}"
    from auto_gptq.utils.import_utils import dynamically_import_QuantLinear as autogptq_getclass_quantlinear
elif args.exllama_path:
    implementation = "exllama"
    sys.path.insert(0, args.exllama_path)
    from model import Ex4bitLinear, ExLlamaConfig

if implementation == "autogptq-triton" and not act_order:
    raise NotImplementedError("autogptq-triton does not support no act-order")

# TODO: not sure about the first one
if implementation == "autogptq-cuda" and act_order is False:
    raise NotImplementedError("autogptq-cuda does not support no act-order")
if implementation == "autogptq-cuda-old" and act_order is True:
    raise NotImplementedError("With autogptq, use [--autogptq-implem cuda] and not [--autogptq-implem cuda-old] when using act-order")

device = torch.device("cuda")

m = args.m
n = args.n
k = args.k

if n % 8 != 0:
    raise ValueError(f"n should be divisible by 8")
if k % args.group_size != 0:
    raise ValueError(f"k should be divisible by group_size={args.group_size}")

tensors = {
    "q_proj.qweight": torch.zeros(n // 8, k, dtype=torch.int32).to(device),
    "q_proj.qzeros": torch.zeros(k // args.group_size, n // 8, dtype=torch.int32).to(device),
    "q_proj.scales": torch.zeros(k // args.group_size, n, dtype=torch.float16).to(device),
    "q_proj.g_idx": torch.zeros(k, dtype=torch.int32).to(device)
}

if implementation == "exllama":
    config = ExLlamaConfig("dummy_config.json")  # actually not used
    if not act_order:
        del tensors["q_proj.g_idx"]
    linear = Ex4bitLinear(config, k, n, has_bias=False, tensors=tensors, key="q_proj")
elif implementation == "gptqforllama":
    linear = GPTQForLlamaQuantLinear(bits, args.group_size, k, n, bias=False)
elif implementation.startswith("autogptq"):
    if implementation == "autogptq-triton":
        use_triton = True
    else:
        use_triton = False

    linear_class = autogptq_getclass_quantlinear(use_triton=use_triton, desc_act=act_order, group_size=args.group_size)

    linear = linear_class(
        bits=bits,
        group_size=args.group_size,
        infeatures=k,
        outfeatures=n,
        bias=False,
    )
elif implementation == "baseline":
    linear = nn.Linear(k, n, bias=False).to(device).to(torch.float16)

# exllama does not inherit from nn.Module
if isinstance(linear, nn.Module):
    linear = linear.eval()
    linear = linear.to(device)

if implementation == "gptqforllama":
    print("Warming-up gptqforllama...")
    autotune_warmup_linear(linear)
elif implementation == "autogptq-triton":
    print("Warming-up autogptq-triton...")
    from auto_gptq.nn_modules.qlinear.qlinear_triton import QuantLinear
    linear.device = device
    QuantLinear.warmup(linear)

inp = torch.rand(1, m, k, dtype=torch.float16).to(device)

print(f"\nInput: {inp.shape}, {inp.dtype}")
print(f"Linear class: {inspect.getfile(linear.__class__)}")
if implementation in ["exllama", "gptqforllama"] or implementation.startswith("autogptq"):
    print(f"Weight: {linear.qweight.shape}, {linear.qweight.dtype}")
    print(f"qzeros: {linear.qzeros.shape}, {linear.qzeros.dtype}")
    print(f"scales: {linear.scales.shape}, {linear.scales.dtype}")
elif implementation == "baseline":
    print(f"Weight: {linear.weight.shape}, {linear.weight.dtype}")

if args.csv and not os.path.isfile("results.csv"):
    header = "m,n,k,implementation,act_order,time_ms_per_op,max_mem_mb\n"
    with open("results.csv", "a") as text_file:
        text_file.write(header)

num_runs = 100
# warmup
with torch.no_grad():
    # Warmup
    if implementation in ["baseline", "gptqforllama"] or implementation.startswith("autogptq"):
        res = linear(inp)
    elif implementation == "exllama":
        res = linear.forward(inp, None)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_event.record()

    if implementation in ["baseline", "gptqforllama"] or implementation.startswith("autogptq"):
        for _ in range(num_runs):
            res = linear(inp)
    elif implementation == "exllama":
        for _ in range(num_runs):
            res = linear.forward(inp, None)

    end_event.record()

    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)

    tps = (start_event.elapsed_time(end_event)) / num_runs
    mem_mb = max_memory * 1e-6

    if args.csv:
        data = ",".join([
            str(m),
            str(n),
            str(k),
            implementation,
            str(act_order),
            f"{tps:.4f}",
            f"{mem_mb:.4f}",
        ])
        with open("results.csv", "a") as text_file:
            text_file.write(data + "\n")

    print(f"\nImplementation: {implementation}")
    if implementation in ["exllama", "gptqforllama"] or implementation.startswith("autogptq"):
        print(f"Act order: {act_order}")
    print(f"Time: {tps:.4f} ms/op")
    print(f"Max memory: {mem_mb:.4f} MB")
