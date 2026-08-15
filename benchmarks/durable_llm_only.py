from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from benchmarks.benchmark import filter_types, load_dataset, parse_entities, parse_target_types
from benchmarks.durable_ginza import _repair_and_load_completed
from src.extractors.llm_only_extractor import FullTextLlmExtractor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Durable, resumable full-text local-LLM benchmark runner"
    )
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument(
        "--output",
        default="results/llm_only_500/chunks/live/per_document.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--think-level",
        choices=("low", "medium", "high"),
        default="medium",
        help="GPT-OSS reasoning effort. The model template defaults to medium.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=4096,
        help="Ollama context length for each request.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    records = load_dataset(Path(cfg["dataset"]["path"]))
    if args.limit is not None:
        records = records[: args.limit]

    target_types = parse_target_types(cfg.get("evaluation", {}).get("target_types"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = _repair_and_load_completed(output)

    remaining = sum(1 for record in records if str(record["id"]) not in completed)
    print(
        f"[durable-llm-only] total={len(records)} completed={len(completed)} "
        f"remaining={remaining} think={args.think_level} num_ctx={args.num_ctx}",
        flush=True,
    )

    extractor = FullTextLlmExtractor(
        model_name=str(cfg["llm"]["model"]),
        think_level=args.think_level,
        num_ctx=args.num_ctx,
    )

    processed_this_run = 0
    with output.open("a", encoding="utf-8", newline="\n", buffering=1) as f:
        for record in records:
            record_id = str(record["id"])
            if record_id in completed:
                continue

            text = str(record["text"])
            truths = filter_types(parse_entities(record.get("entities", [])), target_types)

            before_calls = extractor.stats.calls
            before_seconds = extractor.stats.seconds
            before_failures = extractor.stats.failures

            started = time.perf_counter()
            predictions = extractor.extract(text)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            predictions = filter_types(predictions, target_types)

            llm_calls = extractor.stats.calls - before_calls
            llm_seconds = extractor.stats.seconds - before_seconds
            llm_failures = extractor.stats.failures - before_failures
            meta = dict(extractor.last_response_meta)

            row = {
                "method": "llm_only",
                "id": record.get("id"),
                "latency_ms": elapsed_ms,
                "llm_calls": llm_calls,
                "llm_seconds": llm_seconds,
                "llm_failures": llm_failures,
                "llm_think_level": meta.get("think_level", args.think_level),
                "llm_num_ctx": meta.get("num_ctx", args.num_ctx),
                "llm_done_reason": meta.get("done_reason"),
                "llm_prompt_eval_count": meta.get("prompt_eval_count"),
                "llm_eval_count": meta.get("eval_count"),
                "llm_thinking_chars": meta.get("thinking_chars"),
                "llm_content_chars": meta.get("content_chars"),
                "ginza_available": False,
                "truth": [entity.to_dict() for entity in truths],
                "predictions": [entity.to_dict() for entity in predictions],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

            completed.add(record_id)
            processed_this_run += 1
            print(
                f"[{len(completed)}/{len(records)}] id={record_id} "
                f"latency_ms={elapsed_ms:.2f} llm_calls={llm_calls} "
                f"llm_seconds={llm_seconds:.2f} failures={llm_failures} "
                f"done={meta.get('done_reason')} eval={meta.get('eval_count')} "
                f"thinking_chars={meta.get('thinking_chars')} "
                f"content_chars={meta.get('content_chars')}",
                flush=True,
            )

    print(
        f"[done] completed={len(completed)}/{len(records)} "
        f"processed_this_run={processed_this_run} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
