## Server Launch Commands

### `vllm`

```bash
python3 -m venv venv-vllm
source venv-vllm/bin/activate
pip install vllm==0.7.3

VLLM_USE_V1=1 vllm serve Qwen/Qwen2.5-0.5B --quantization fp8 --disable-log-requests
```

### `sglang`

```bash
docker run --gpus all --shm-size 32g -p 8000:8000 -v ~/.cache/huggingface:/root/.cache/huggingface --env "HF_TOKEN=$HF_TOKEN" --ipc=host lmsysorg/sglang:latest python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B --tp-size 1 --dp-size 1 --disable-cuda-graph --quantization fp8 --enable-torch-compile --sampling-backend flashinfer --attention-backend flashinfer --port 8000
```

## Benchmark Clients

The following results were generated on H100:

### Installation

```bash
git clone https://github.com/robertgshaw2-redhat/benchmark-single-token.git
cd benchmark-single-token
pip install -r requirements.txt
```

### Benchmark Offline Throughput

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

### Benchmark Online Serving With 10 Concurrent Requests

```bash
python3 benchmark.py --max-concurrency 10
```

- `sglang` output:

```bash
Total Requests:  100
Prompts/Request:  250
Total Time:  118.89
Avg Latency:  11.29
Min Latency:  2.58
Max Latency:  14.27
```

- `vllm` output:

```bash
Total Requests:  100
Prompts/Request:  250
Total Time:  57.20
Avg Latency:  5.50
Min Latency:  1.33
Max Latency:  8.01
```
