#!/usr/bin/env python3
"""
Decode student rollout telemetry and run basic consistency checks.

This script:
- reads a telemetry_rollouts.jsonl file
- loads model_name and renderer_name from the sibling config.json
- loads the matching tokenizer
- decodes the first N rollout token sequences
- checks tensor length consistency for saved telemetry fields
- runs a lightweight renderer-format check for role_colon outputs
- marks which decoded rollouts contain \\boxed{}
- saves full text for rollouts with boxed answers
- writes a small markdown report for manual inspection
"""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parent


DEFAULT_TELEMETRY = Path(
    "/Users/dudberoll/tinker-examples/distillation/"
    "onpolicy-deepmath-g64-gs2-s50-20260304-114700/telemetry_rollouts.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode and validate rollout telemetry.")
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=DEFAULT_TELEMETRY,
        help="Path to telemetry_rollouts.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many rollout records to decode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional markdown report path. Defaults to <project>/decoded_rollouts_check.md",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json next to telemetry: {config_path}")
    return json.loads(config_path.read_text())


def find_local_hf_snapshot(model_name: str) -> Path | None:
    hub_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir_name = "models--" + model_name.replace("/", "--")
    snapshots_dir = hub_root / model_dir_name / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
    if not snapshots:
        return None
    return snapshots[-1]


def load_tokenizer(model_name: str):
    local_snapshot = find_local_hf_snapshot(model_name)
    source = str(local_snapshot) if local_snapshot is not None else model_name
    return AutoTokenizer.from_pretrained(source, use_fast=True, local_files_only=True)


def parse_with_renderer_name(tokenizer, renderer_name: str, tokens: list[int]) -> tuple[str, bool, str]:
    eos_token_id = tokenizer.eos_token_id
    raw_tokens = list(tokens)
    terminated_with_eos = False
    if eos_token_id is not None and raw_tokens and raw_tokens[-1] == eos_token_id:
        raw_tokens = raw_tokens[:-1]
        terminated_with_eos = True

    decoded = tokenizer.decode(raw_tokens)

    if renderer_name == "role_colon":
        parts = decoded.split("\n\nUser:")
        if len(parts) == 1:
            return decoded.strip(), False, "missing_stop_delimiter"
        if len(parts) == 2:
            return parts[0].strip(), not terminated_with_eos, "ok" if not terminated_with_eos else "terminated_with_eos"
        return parts[0].strip(), False, "multiple_stop_delimiters"

    # Fallback for other renderers: decode-only check
    return decoded.strip(), True, f"decode_only_renderer_{renderer_name}"


def is_suspicious_text(text: str) -> tuple[bool, dict[str, float]]:
    if not text:
        return False, {"non_printable_ratio": 0.0, "punct_ratio": 0.0, "replacement_char_count": 0.0}

    printable = set(string.printable)
    non_printable_count = sum(ch not in printable and not ch.isspace() for ch in text)
    punct_count = sum(ch in string.punctuation for ch in text)
    replacement_char_count = text.count("\ufffd")

    non_printable_ratio = non_printable_count / len(text)
    punct_ratio = punct_count / len(text)

    suspicious = (
        replacement_char_count > 0
        or non_printable_ratio > 0.10
        or punct_ratio > 0.60
    )
    return suspicious, {
        "non_printable_ratio": non_printable_ratio,
        "punct_ratio": punct_ratio,
        "replacement_char_count": float(replacement_char_count),
    }


def normalize_preview(text: str, max_chars: int = 1200) -> str:
    text = text.replace("\r", "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def contains_boxed_answer(text: str) -> bool:
    return "\\boxed{" in text or "\\boxed " in text


def extract_completion_text(decoded_text: str, renderer_name: str) -> str:
    if renderer_name == "role_colon":
        marker = "Assistant:"
        if marker in decoded_text:
            return decoded_text.rsplit(marker, 1)[-1].strip()
    return decoded_text.strip()


def check_lengths(record: dict[str, Any]) -> tuple[bool, dict[str, int]]:
    lengths = {
        "tokens": len(record["tokens"]),
        "mask": len(record["mask"]),
        "logp_student": len(record["logp_student"]),
        "logp_teacher": len(record["logp_teacher"]),
        "step_kl": len(record["step_kl"]),
        "advantages": len(record["advantages"]),
    }
    values = list(lengths.values())
    return len(set(values)) == 1, lengths


def load_records(path: Path, limit: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            records.append(json.loads(line))
    return records


def build_report(
    telemetry_path: Path,
    config: dict[str, Any],
    decoded_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    full_text_dir: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Rollout Decode Check")
    lines.append("")
    lines.append(f"- Telemetry: `{telemetry_path}`")
    lines.append(f"- Model: `{config['model_name']}`")
    lines.append(f"- Renderer: `{config['renderer_name']}`")
    lines.append(f"- Decoded records: `{summary['decoded_records']}`")
    lines.append(f"- Length-consistent records: `{summary['length_ok_count']}/{summary['decoded_records']}`")
    lines.append(f"- Renderer parse_success: `{summary['parse_success_count']}/{summary['decoded_records']}`")
    lines.append(f"- Suspicious decoded text: `{summary['suspicious_count']}/{summary['decoded_records']}`")
    lines.append(f"- Records with `\\boxed{{}}`: `{summary['boxed_count']}/{summary['decoded_records']}`")
    lines.append(f"- Full boxed texts dir: `{full_text_dir}`")
    lines.append("")

    for row in decoded_rows:
        lines.append(f"## Rollout {row['index']}")
        lines.append("")
        lines.append(f"- length_ok: `{row['length_ok']}`")
        lines.append(f"- parse_success: `{row['parse_success']}`")
        lines.append(f"- parse_status: `{row['parse_status']}`")
        lines.append(f"- suspicious_text: `{row['suspicious_text']}`")
        lines.append(f"- contains_boxed: `{row['contains_boxed']}`")
        lines.append(f"- token_count: `{row['token_count']}`")
        lines.append(f"- active_token_count: `{row['active_token_count']}`")
        lines.append(f"- h_student: `{row['h_student']:.6f}`")
        lines.append(f"- h_teacher: `{row['h_teacher']:.6f}`")
        lines.append(f"- abs_h_diff: `{row['abs_h_diff']:.6f}`")
        lines.append(f"- individual_reward: `{row['individual_reward']}`")
        lines.append(f"- non_printable_ratio: `{row['non_printable_ratio']:.4f}`")
        lines.append(f"- punct_ratio: `{row['punct_ratio']:.4f}`")
        lines.append(f"- replacement_char_count: `{int(row['replacement_char_count'])}`")
        if row["contains_boxed"]:
            lines.append(f"- full_text_file: [{row['full_text_file_name']}]({row['full_text_file_path']})")
        lines.append("")
        lines.append("### Field lengths")
        lines.append("")
        for key, value in row["lengths"].items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
        lines.append("### Decoded Text")
        lines.append("")
        lines.append("```text")
        lines.append(normalize_preview(row["decoded_text"]))
        lines.append("```")
        lines.append("")
        lines.append("### Renderer Parsed Text")
        lines.append("")
        lines.append("```text")
        lines.append(normalize_preview(row["parsed_text"]))
        lines.append("```")
        lines.append("")
        lines.append("### Completion Text")
        lines.append("")
        lines.append("```text")
        lines.append(normalize_preview(row["completion_text"]))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    telemetry_path = args.telemetry.resolve()
    run_dir = telemetry_path.parent
    config = load_config(run_dir)

    tokenizer = load_tokenizer(config["model_name"])

    records = load_records(telemetry_path, args.limit)
    decoded_rows: list[dict[str, Any]] = []

    length_ok_count = 0
    parse_success_count = 0
    suspicious_count = 0
    boxed_count = 0
    full_text_dir = REPO_ROOT / "decoded_rollouts_full_texts"
    full_text_dir.mkdir(exist_ok=True)

    for index, record in enumerate(records):
        length_ok, lengths = check_lengths(record)
        if length_ok:
            length_ok_count += 1

        tokens = record["tokens"]
        decoded_text = tokenizer.decode(tokens)
        completion_text = extract_completion_text(decoded_text, config["renderer_name"])
        parsed_text, parse_success, parse_status = parse_with_renderer_name(
            tokenizer, config["renderer_name"], tokens
        )
        if parse_success:
            parse_success_count += 1

        suspicious_text, stats = is_suspicious_text(decoded_text)
        if suspicious_text:
            suspicious_count += 1
        contains_boxed = contains_boxed_answer(completion_text)
        full_text_file_name = ""
        full_text_file_path = ""
        if contains_boxed:
            boxed_count += 1
            full_text_file = full_text_dir / f"rollout_{index:03d}_boxed.txt"
            full_text_file.write_text(completion_text)
            full_text_file_name = full_text_file.name
            full_text_file_path = str(full_text_file)

        row = {
            "index": index,
            "length_ok": length_ok,
            "lengths": lengths,
            "parse_success": parse_success,
            "parse_status": parse_status,
            "suspicious_text": suspicious_text,
            "token_count": len(tokens),
            "active_token_count": sum(1 for x in record["mask"] if x > 0),
            "h_student": float(record.get("h_student", 0.0)),
            "h_teacher": float(record.get("h_teacher", 0.0)),
            "abs_h_diff": abs(float(record.get("h_student", 0.0)) - float(record.get("h_teacher", 0.0))),
            "individual_reward": record.get("individual_reward"),
            "decoded_text": decoded_text,
            "completion_text": completion_text,
            "parsed_text": parsed_text,
            "contains_boxed": contains_boxed,
            "full_text_file_name": full_text_file_name,
            "full_text_file_path": full_text_file_path,
            **stats,
        }
        decoded_rows.append(row)

    summary = {
        "decoded_records": len(decoded_rows),
        "length_ok_count": length_ok_count,
        "parse_success_count": parse_success_count,
        "suspicious_count": suspicious_count,
        "boxed_count": boxed_count,
    }

    output_path = args.output or (REPO_ROOT / "decoded_rollouts_check.md")
    report = build_report(telemetry_path, config, decoded_rows, summary, full_text_dir)
    output_path.write_text(report)

    print(f"Telemetry: {telemetry_path}")
    print(f"Model: {config['model_name']}")
    print(f"Renderer: {config['renderer_name']}")
    print(f"Decoded records: {summary['decoded_records']}")
    print(f"Length-consistent: {summary['length_ok_count']}/{summary['decoded_records']}")
    print(f"Renderer parse_success: {summary['parse_success_count']}/{summary['decoded_records']}")
    print(f"Suspicious decoded text: {summary['suspicious_count']}/{summary['decoded_records']}")
    print(f"Contains boxed: {summary['boxed_count']}/{summary['decoded_records']}")
    print(f"Full boxed texts dir: {full_text_dir}")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
