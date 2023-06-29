This repository aims to compare the available open-source GEMM / GEMV kernels using a mixed precision scheme int4 / fp16.

## Available implementations

- [x] https://github.com/qwopqwop200/GPTQ-for-LLaMa
- [ ] https://github.com/NVIDIA/FasterTransformer (only per-channel quantization, per-block not open-sourced)
- [x] https://github.com/turboderp/exllama
- [ ] https://github.com/PanQiWei/AutoGPTQ

## Results


## Run the benchmark

A=m*k, B=k*n, compute C= A*B^T

First lock GPU frequency.

Then, set up conda environments named `gptqforllama`, `exllama` satisfying each repo requirements (one requires torch stable, the other nightly).

Run exllama in `exllama` env:
```
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 4 --exllama-path ../exllama
```

Run gptqforllama in `gptqforllama` env:
```
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 4 --gptqforllama-path ../GPTQ-for-LLaMa
```

Run PyTorch fp16 * fp16 baseline:
```
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 4 --baseline
```
