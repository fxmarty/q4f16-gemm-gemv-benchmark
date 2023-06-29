eval "$(conda shell.bash hook)"

conda activate quant # this is gptqforllama, autogptq
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --baseline --csv

CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --gptqforllama-path ../GPTQ-for-LLaMa/ --act-order yes --csv

CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --autogptq-path ../AutoGPTQ/ --autogptq-implem triton --act-order yes --csv
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --autogptq-path ../AutoGPTQ/ --autogptq-implem cuda-old --act-order no --csv
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --autogptq-path ../AutoGPTQ/ --autogptq-implem cuda --act-order yes --csv

conda activate exllama
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --exllama-path ../exllama/ --act-order no --csv
CUDA_VISIBLE_DEVICES=0 python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --exllama-path ../exllama/ --act-order yes --csv


cat results.csv
rm results.csv
