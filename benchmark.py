import aiohttp
import asyncio
import json
import time
import numpy as np

from argparse import ArgumentParser as FlexibleArgumentParser
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from typing import List, Optional

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

def sample_random_requests(
    input_len: int,
    num_prompts: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[str]:

    tokens = np.random.randint(0, tokenizer.vocab_size, size=(num_prompts + input_len))
    prompts = [
        tokenizer.decode(tokens[i:i+input_len]) for i in range(0, num_prompts)
    ]
    assert len(prompts) == num_prompts

    return prompts

async def limited_request_task(url: str,
                               model: str,
                               prompts: List[str],
                               output_len: int,
                               semaphore: Optional[asyncio.Semaphore] = None):
    if semaphore:
        async with semaphore:
            return await request_task(url, model, prompts, output_len)
    else:
        return await request_task(url, model, prompts, output_len)

async def request_task(url: str,
                       model: str,
                       prompts: List[str],
                       output_len: int):

    payload = {
        "model": model,
        "prompt": prompts,
        "temperature": 0.0,
        "max_tokens": output_len,
    }
    headers = {
        "Authorization": "Bearer OSS_NEEDS_NO_TOKEN"
    }

    async with aiohttp.ClientSession(trust_env=True,
                                     timeout=AIOHTTP_TIMEOUT) as session:
        start = time.perf_counter()
        async with session.post(url=url, json=payload, headers=headers) as response:
            assert response.status == 200
            async for chunk_bytes in response.content:
                result = json.loads(chunk_bytes)
        end = time.perf_counter()
        latency = end - start

    return result, latency


async def make_requests(url: str,
                        model: str,
                        prompts_grouped: List[List[str]],
                        output_len: int,
                        max_concurrency: float) -> None:
    
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
    
    tasks = []
    start = time.perf_counter()
    for prompts in prompts_grouped:
        tasks.append(asyncio.create_task(limited_request_task(
            url, model, prompts, output_len, semaphore)))
    
    results = await asyncio.gather(*tasks)
    end = time.perf_counter()

    print("Total Requests: ", len(results))
    total_generations = 0
    latencies = []
    for result, latency in results:
        total_generations += len(result["choices"])
        latencies.append(latency)
    print("Prompts/Request: ", total_generations // len(results))
    print(f"Total Time: {end - start: 0.2f}")
    print(f"Avg Latency: {np.average(latencies): 0.2f}")
    print(f"Min Latency: {np.amin(latencies): 0.2f}")
    print(f"Max Latency: {np.amax(latencies): 0.2f}")
    

if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--num-reqs", type=int, default=100)
    parser.add_argument("--num-prompts-per-req", type=int, default=250)
    parser.add_argument("--input-len", type=int, default=600)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=None)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    total_prompts = args.num_reqs * args.num_prompts_per_req

    # Sample random requests.
    print("Sampling random requests [this could take some time]...")
    prompts = sample_random_requests(
        input_len=args.input_len,
        num_prompts=total_prompts,
        tokenizer=tokenizer,
    )

    # Split random requests into groups of num_prompts_per_req.
    print("Splitting into groups of prompts...")
    prompts_grouped = [
        prompts[i:i + args.num_prompts_per_req]
        for i in range(0, total_prompts, args.num_prompts_per_req)
    ]
    assert len(prompts_grouped) == args.num_reqs

    print("Making requests...")
    asyncio.run(make_requests(
        url=f"http://{args.host}:{args.port}/v1/completions",
        model=args.model,
        prompts_grouped=prompts_grouped,
        output_len=args.output_len,
        max_concurrency=args.max_concurrency)
    )
