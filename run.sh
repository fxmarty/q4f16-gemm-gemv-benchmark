eval "$(conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=0

conda activate quant # this is gptqforllama, autogptq
python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --baseline --csv

python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --gptqforllama-path ../GPTQ-for-LLaMa/ --act-order yes --csv

python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --autogptq-path ../AutoGPTQ/ --autogptq-implem triton --act-order yes --csv
python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --autogptq-path ../AutoGPTQ/ --autogptq-implem cuda-old --act-order no --csv
python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --autogptq-path ../AutoGPTQ/ --autogptq-implem cuda --act-order yes --csv

conda activate exllama
python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --exllama-path ../exllama/ --act-order no --csv
python run_benchmark.py --m 1 --n 8192 --k 8192 --group_size 128 --exllama-path ../exllama/ --act-order yes --csv

cat results.csv
rm results.csv
