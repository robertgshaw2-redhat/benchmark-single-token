import asyncio
import numpy as np

from openai import AsyncOpenAI
from argparse import ArgumentParser as FlexibleArgumentParser
from transformers import PreTrainedTokenizerBase, AutoTokenizer
from typing import List, Tuple


def sample_random_requests(
    prefix_len: int,
    input_len: int,
    num_prompts: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
) -> List[str]:
    prefix_token_ids = np.random.randint(0,
                                         tokenizer.vocab_size,
                                         size=prefix_len).tolist()

    input_lens = np.random.randint(
        int(input_len * range_ratio),
        input_len + 1,
        size=num_prompts,
    )
    offsets = np.random.randint(0, tokenizer.vocab_size, size=num_prompts)
    prompts = []
    for i in range(num_prompts):
        prompts.append(tokenizer.decode(prefix_token_ids +
                                  [(offsets[i] + i + j) % tokenizer.vocab_size
                                   for j in range(input_lens[i])]))

    return prompts

async def request_task(client: AsyncOpenAI,
                       prompts: List[str],
                       output_len: int):
    return await client.completions.create(prompts=prompts,
                                           max_tokens=output_len,
                                           temperature=0.0)

async def make_requests(client: AsyncOpenAI,
                        prompts_grouped: List[List[str]],
                        output_len: int,
                        request_rate: float) -> None:
    
    tasks = []
    for prompts in prompts_grouped:
        tasks.append(asyncio.create_task(request_task(client, prompts, output_len)))
        if request_rate != float("inf"):
            await asyncio.sleep(1 / request_rate)
    
    results = await asyncio.gather(*tasks)
    print(results)

if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-reqs", type=int, default=1000)
    parser.add_argument("--num-prompts-per-req", type=int, default=1000)
    parser.add_argument("--input-len", type=int, default=600)
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    total_prompts = args.num_reqs * args.num_prompts_per_req

    # Sample random requests.
    prompts = sample_random_requests(
        prefix_len=0,
        input_len=args.input_len,
        num_prompts=total_prompts,
        range_ratio=1.0,
        tokenizer=tokenizer,
    )

    # Split random requests into groups of num_prompts_per_req.
    prompts_grouped = [
        prompts[i:i + args.num_prompts_per_req]
        for i in range(0, total_prompts, args.num_prompts_per_req)
    ]
    assert len(prompts_grouped) == args.num_reqs


    client = AsyncOpenAI(
        api_key="OPEN_SOURCE_WINS",
        base_url=f"http://{args.host}:{args.port}",
    )

    asyncio.run(make_requests(client, prompts_grouped, args.output_len, args.request_rate))
