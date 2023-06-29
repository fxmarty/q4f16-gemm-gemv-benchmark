import torch
import torch.nn as nn
import argparse
import sys

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
    type=int,
    default=None,
    help="Path to exllama repo",
)
implem_group.add_argument(
    "--baseline",
    action="store_true"
    help="Use PyTorch fp16 x fp16",
)
implem_group.add_argument(
    "--act-order",
    action="store_true"
    help="Use act-order option (that reorders the groups?)",
)

hidden_size = 8192
bits = 4

args = parser.parse_args()

if args.act_order:
    raise NotImplementedError("act_order to be implemented")

# Please pardon me for this ugliness :)
if args.baseline:
    implementation = "baseline"
elif args.gptqforllama_path:
    implementation = "gptqforllama"
    sys.path.insert(0, args.gptqforllama_path)
    from quant.quant_linear import QuantLinear as GPTQForLlamaQuantLinear
    from quant.quant_linear import autotune_warmup_linear
elif args.exllama_path:
    implementation = "exllama"
    sys.path.insert(0, args.exllama_path)
    from model import Ex4bitLinear as Exllama4bitLinear

device = torch.device("cuda")

tensors = {
    "q_proj.qweight": torch.zeros(n // 8, k, dtype=torch.int32).to(device),
    "q_proj.qzeros": torch.zeros(k // args.group_size, n // 8, dtype=torch.int32).to(device),
    "q_proj.scales": torch.zeros(k // args.group_size, n, dtype=torch.float16).to(device),
    "q_proj.g_idx": torch.zeros(k, dtype=torch.int32).to(device)
}

if args.exllama_path:
    linear = Exllama4bitLinear(config, hidden_size, num_attention_heads * head_dim, False, tensors, "q_proj")

elif args.gptqforllama_path:
    linear = QuantLinear(bits, args.group_size, k, n, bias=False)

    # TODO: probably useless
    linear.qweight = tensors["q_proj.qweight"]
    linear.qzeros = tensors["q_proj.qzeros"]
    linear.scales = tensors["q_proj.scales"]
    linear.g_idx = tensors["q_proj.g_idx"]
elif args.baseline:
    linear = nn.Linear(k, n, bias=False).to(device).to(torch.float16)

linear = linear.eval()
linear = linear.to(device)

if args.gptqforllama_path:
    autotune_warmup_linear(q4linear)

inp = torch.rand(1, 1, k, dtype=torch.float16).to(device)

num_runs = 100
# warmup
with torch.no_grad():
    # Warmup
    if args.gptqforllama_path or args.baseline:
        res = linear(inp)
    elif args.exllama_path:
        res = linear.forward(inp, None)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_event.record()

    if args.gptqforllama_path or args.baseline:
        for _ in range(num_runs):
            res = q4linear(inp)
    elif args.exllama_path:
        for _ in range(num_runs):
            res = q4linear.forward(inp, None)

    end_event.record()

    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device)

    tps = (start_event.elapsed_time(end_event) * 1.0e-3) / num_runs
    mem_mb = max_memory * 1e-6

    print(f"Time ({implementation}):", tps)
    print(f"Max memory ({implementation}, MB):", mem_mb)
