#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json

from dotenv import load_dotenv
import tinker
from tinker import types
from tinker_cookbook.tokenizer_utils import get_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test loading a Tinker checkpoint.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--sampler-path", default=None)
    parser.add_argument("--sample-prompt", default="2+2=")
    parser.add_argument("--sample-max-tokens", type=int, default=16)
    parser.add_argument("--base-url", default=None)
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    load_dotenv("/Users/dudberoll/prj/diploma/.env")

    service_client = tinker.ServiceClient(base_url=args.base_url)
    result: dict[str, object] = {
        "base_model": args.base_model,
        "state_path": args.state_path,
        "sampler_path": args.sampler_path,
    }

    tokenizer = get_tokenizer(args.base_model)

    if args.state_path:
        training_client = await service_client.create_lora_training_client_async(
            args.base_model, rank=128
        )
        future = await training_client.load_state_with_optimizer_async(args.state_path)
        _ = await future.result_async()
        training_tokenizer = training_client.get_tokenizer()
        result["training_state_loaded"] = True
        result["training_tokenizer_vocab_size"] = getattr(training_tokenizer, "vocab_size", None)
    else:
        result["training_state_loaded"] = False

    if args.sampler_path:
        sampling_client = service_client.create_sampling_client(
            model_path=args.sampler_path,
            base_model=args.base_model,
        )
        prompt = types.ModelInput.from_ints(tokenizer.encode(args.sample_prompt))
        sample = await sampling_client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=types.SamplingParams(
                max_tokens=args.sample_max_tokens,
                temperature=0.0,
            ),
        )
        seq = sample.sequences[0]
        result["sampling_checkpoint_loaded"] = True
        result["sample_num_tokens"] = len(seq.tokens)
        result["sample_text"] = tokenizer.decode(seq.tokens)
    else:
        result["sampling_checkpoint_loaded"] = False

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
