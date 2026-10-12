#!/usr/bin/env python3
"""
Evaluate a model on a fixed DeepMath subset and save both the subset and results.

This script:
1. Creates or reuses a frozen DeepMath evaluation subset.
2. Samples model responses with the same renderer/format expectations used in training.
3. Grades each response with the same MathEnv logic (`correct`, `format`, reward).
4. Saves per-example results incrementally plus a final summary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
import tinker
from tinker import types

from tinker_cookbook import renderers
from tinker_cookbook.recipes.math_rl.math_env import MathEnv
from tinker_cookbook.tokenizer_utils import get_tokenizer


@dataclass
class EvalItem:
    eval_index: int
    source_index: int
    question: str
    final_answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B-Base")
    parser.add_argument("--sampler-path", default=None)
    parser.add_argument("--tokenizer-model-name", default=None)
    parser.add_argument("--renderer-name", default="role_colon")
    parser.add_argument("--subset-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--subset-path",
        default="/Users/dudberoll/prj/diploma/eval_subsets/deepmath_eval_512_seed0.jsonl",
    )
    parser.add_argument(
        "--results-dir",
        default="/Users/dudberoll/prj/diploma/eval_runs/qwen3_8b_base_deepmath_512_seed0",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_or_create_subset(subset_path: Path, subset_size: int, seed: int) -> list[EvalItem]:
    ensure_parent(subset_path)
    if subset_path.exists():
        items = []
        with subset_path.open() as f:
            for line in f:
                if line.strip():
                    items.append(EvalItem(**json.loads(line)))
        return items

    ds = load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(seed=seed)
    items: list[EvalItem] = []
    for source_index, row in enumerate(ds):  # type: ignore
        question = row.get("question", "")
        final_answer = row.get("final_answer", "")
        if not question or not final_answer:
            continue
        items.append(
            EvalItem(
                eval_index=len(items),
                source_index=source_index,
                question=question,
                final_answer=final_answer,
            )
        )
        if len(items) >= subset_size:
            break

    with subset_path.open("w") as f:
        for item in items:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    return items


async def sample_one(
    item: EvalItem,
    sampling_client: tinker.SamplingClient,
    renderer_name: str,
    tokenizer: Any,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)
    env = MathEnv(item.question, item.final_answer, renderer)
    observation, stop_condition = await env.initial_observation()
    sampling_params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_condition,
    )
    result = await sampling_client.sample_async(
        prompt=observation,
        num_samples=1,
        sampling_params=sampling_params,
    )
    seq = result.sequences[0]
    response_tokens = list(seq.tokens)
    message, parse_success = renderer.parse_response(response_tokens)
    content = renderers.get_text_content(message)
    step_result = await env.step(response_tokens)

    return {
        "eval_index": item.eval_index,
        "source_index": item.source_index,
        "question": item.question,
        "reference_answer": item.final_answer,
        "response_text": content,
        "response_tokens": response_tokens,
        "num_response_tokens": len(response_tokens),
        "parse_success": bool(parse_success),
        "format": float(step_result.metrics.get("format", 0.0)),
        "correct": float(step_result.metrics.get("correct", 0.0)),
        "reward": float(step_result.reward),
    }


async def evaluate(args: argparse.Namespace) -> None:
    load_dotenv("/Users/dudberoll/prj/diploma/.env")
    subset_path = Path(args.subset_path)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.jsonl"
    summary_path = results_dir / "summary.json"

    subset = load_or_create_subset(subset_path, args.subset_size, args.seed)

    service_client = tinker.ServiceClient()
    if args.sampler_path:
        sampling_client = service_client.create_sampling_client(model_path=args.sampler_path)
    else:
        sampling_client = service_client.create_sampling_client(base_model=args.model_name)
    tokenizer_model_name = args.tokenizer_model_name or args.model_name
    tokenizer = get_tokenizer(tokenizer_model_name)

    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    completed = 0
    collected: list[dict[str, Any]] = []

    if results_path.exists():
        results_path.unlink()

    async def run_item(item: EvalItem) -> dict[str, Any]:
        nonlocal completed
        async with semaphore:
            record = await sample_one(
                item=item,
                sampling_client=sampling_client,
                renderer_name=args.renderer_name,
                tokenizer=tokenizer,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            async with write_lock:
                with results_path.open("a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            completed += 1
            if completed % 25 == 0 or completed == len(subset):
                print(f"Completed {completed}/{len(subset)}")
            return record

    records = await asyncio.gather(*[run_item(item) for item in subset])
    records = sorted(records, key=lambda x: x["eval_index"])
    collected.extend(records)

    corrects = [r["correct"] for r in collected]
    formats = [r["format"] for r in collected]
    rewards = [r["reward"] for r in collected]
    lengths = [r["num_response_tokens"] for r in collected]
    parse_successes = [1.0 if r["parse_success"] else 0.0 for r in collected]
    boxed_rate = mean(1.0 if "\\boxed{" in r["response_text"] else 0.0 for r in collected)
    maxed_out_rate = mean(1.0 if r["num_response_tokens"] >= args.max_tokens else 0.0 for r in collected)

    summary = {
        "model_name": args.model_name,
        "sampler_path": args.sampler_path,
        "tokenizer_model_name": tokenizer_model_name,
        "renderer_name": args.renderer_name,
        "subset_path": str(subset_path),
        "subset_size": len(subset),
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "correct_mean": mean(corrects),
        "format_mean": mean(formats),
        "reward_mean": mean(rewards),
        "parse_success_mean": mean(parse_successes),
        "boxed_rate": boxed_rate,
        "num_response_tokens_mean": mean(lengths),
        "num_response_tokens_median": median(lengths),
        "maxed_out_rate": maxed_out_rate,
        "results_path": str(results_path),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    asyncio.run(evaluate(args))


if __name__ == "__main__":
    main()
