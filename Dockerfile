FROM nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV PATH="/home/user/miniconda3/bin:${PATH}"
ARG PATH="/home/user/miniconda3/bin:${PATH}"

ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

USER user
WORKDIR /home/user

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir .conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda init bash

WORKDIR /home/user
RUN git clone https://github.com/PanQiWei/AutoGPTQ.git
RUN git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git
RUN git clone https://github.com/turboderp/exllama.git

RUN conda create -n exllama python=3.9 -y
RUN conda create -n quant python=3.9 -y

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH

RUN eval "$(conda shell.bash hook)" \
    && conda activate exllama \
    && pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118 \
    && pip install numpy \
    && pip install safetensors sentencepiece ninja

RUN eval "$(conda shell.bash hook)" \
    && conda activate quant \
    && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \
    && cd GPTQ-for-LLaMa && pip install -r requirements.txt \
    && cd ../AutoGPTQ && pip install -e .

RUN git clone https://github.com/fxmarty/q4f16-gemm-gemv-benchmark.git
WORKDIR /home/user/q4f16-gemm-gemv-benchmark
CMD CUDA_VISIBLE_DEVICES=0 bash run.sh
