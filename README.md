This repository aims to compare the available open-source GEMM / GEMV kernels using a mixed precision scheme int4 / fp16, with per-group quantization.

## Available implementations

- [x] https://github.com/qwopqwop200/GPTQ-for-LLaMa
- [x] https://github.com/turboderp/exllama
- [x] https://github.com/PanQiWei/AutoGPTQ
- [ ] https://github.com/NVIDIA/FasterTransformer (only per-channel quantization, per-block not open-sourced, so not compared but sounds promising as based on CUTLASS)
- [ ] AWQ implem https://github.com/mit-han-lab/llm-awq/tree/main/awq/kernels
- [ ] Probably missing others

## Results

On A100-SXM4-80GB & Intel Xeon Platinum 8275CL CPU + CUDA 11.7/11.8 (should be rerun in docker):

|m  |n   |k   |implementation|act_order        |Time (ms/op)|Max mem (MB)|
|---|----|----|--------------|-----------------|--------------|----------|
|1  |8192|8192|baseline      |True             |0.0937        |177.6845  |
|1  |8192|8192|gptqforllama  |True             |0.2038        |69.8450   |
|1  |8192|8192|exllama       |False            |0.0681        |34.9143   |
|1  |8192|8192|exllama       |True             |0.0675        |34.9471   |
|1  |8192|8192|autogptq-triton|True             |0.3990        |69.8450   |
|1  |8192|8192|autogptq-cuda-old|False            |0.0831        |71.9585   |
|1  |8192|8192|autogptq-cuda |True             |0.1546        |69.8778   |

On RTX 4090 + AMD Ryzen 9 7950X CPU + CUDA 11.8:

TODO

On A10G + AMD EPYC 7R32 CPU + CUDA 11.8 (docker):

|m  |n   |k   |implementation|act_order        |Time (ms/op)|Max mem (MB)|
|---|----|----|--------------|-----------------|--------------|----------|
|1  |8192|8192|baseline      |True             |0.2891        |177.6845  |
|1  |8192|8192|gptqforllama  |True             |0.1746        |69.8450   |
|1  |8192|8192|autogptq-triton|True             |0.2963        |69.8450   |
|1  |8192|8192|autogptq-cuda-old|False            |0.0979        |71.9585   |
|1  |8192|8192|autogptq-cuda |True             |0.1483        |69.8778   |
|1  |8192|8192|exllama       |False            |0.0842        |34.9143   |
|1  |8192|8192|exllama       |True             |0.0839        |34.9471   |

## Run the benchmark

A=m * k, B=k * n, compute C= A*B^T

It can be a good idea to first lock the GPU frequency, see https://github.com/NVIDIA/cutlass/issues/430#issuecomment-1069535238

Run exllama in `exllama` env:
```
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --exllama-path ../exllama --act-order yes
```

Run gptqforllama in `gptqforllama` env:
```
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --gptqforllama-path ../GPTQ-for-LLaMa --act-order yes
```

Run AutoGPTQ (specify `--autogptq-implem {triton, cuda-old, cuda}`):
```
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --autogptq-path ../AutoGPTQ/ --autogptq-implem triton --act-order yes
```

Run PyTorch fp16 * fp16 baseline:
```
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --baseline
```

## Run all benchmarks

Follow https://stackoverflow.com/a/61737404 and

```
docker build -f Dockerfile --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t container-q4f16 .
```

and

```
docker run --gpus device=0 -it --rm container-q4f16:latest /bin/bash run.sh
```
