# Launch Servers

## vLLM Launch

```bash
python3 -m venv venv-vllm
source venv-vllm/bin/activate
pip install vllm==0.7.3

VLLM_USE_V1=1 vllm serve Qwen/Qwen2.5-0.5B --disable-log-requests
```

## Sglang Launch

```bash
python3 -m venv venv-sglang
source venv-sglang/bin/activate
pip install "sglang[all]==0.4.3.post2" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

python -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B
```

# Benchmark Code

```bash
pip install -r requirements.txt
```

## Benchmark Offline Throughput

The following results were generated on H100:

```bash
python3 benchmark.py
```

- `sglang` output:

```bash
Total Requests:  100
Prompts/Request:  250
Total Time:  145.07
Avg Latency:  77.86
Min Latency:  36.21
Max Latency:  144.99
```

- `vllm` output:

```bash
Total Requests:  100
Prompts/Request:  250
Total Time:  51.13
Avg Latency:  27.50
Min Latency:  1.95
Max Latency:  50.86
```

## Benchmark Online Serving With 10 Concurrency Requests

```bash
python3 benchmark.py --port 8001 --max-concurrency 10
```

- `sglang` output:

```bash
Total Requests:  100
Prompts/Request:  250
Total Time:  123.09
Avg Latency:  11.66
Min Latency:  3.48
Max Latency:  14.85
```

- `vllm` output:

```bash

```


