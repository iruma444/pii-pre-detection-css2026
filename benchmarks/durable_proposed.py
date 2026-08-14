from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from benchmarks.benchmark import filter_types, load_dataset, parse_entities, parse_target_types
from benchmarks.durable_ginza import _repair_and_load_completed
from src.pipeline import PiiPipeline
from src.schemas import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Durable, resumable proposed-method benchmark runner"
    )
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument(
        "--output",
        default="results/proposed_500/chunks/live/per_document.jsonl",
    )
    parser.add_argument("--limit", type=int, default=None)
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
        f"[durable-proposed] total={len(records)} completed={len(completed)} "
        f"remaining={remaining}",
        flush=True,
    )

    pipeline = PiiPipeline(
        PipelineConfig(
            use_regex=True,
            use_dict=True,
            use_nlp=True,
            use_llm=True,
            llm_model=str(cfg["llm"]["model"]),
        )
    )
    if pipeline.ginza_available is not True:
        raise SystemExit("ja_ginza is not available; refusing to record proposed benchmark")
    if pipeline.llm_refiner is None:
        raise SystemExit("LLM refiner is not available; refusing to record proposed benchmark")

    processed_this_run = 0
    with output.open("a", encoding="utf-8", newline="\n", buffering=1) as f:
        for record in records:
            record_id = str(record["id"])
            if record_id in completed:
                continue

            text = str(record["text"])
            truths = filter_types(parse_entities(record.get("entities", [])), target_types)

            before_calls = pipeline.llm_refiner.stats.calls
            before_seconds = pipeline.llm_refiner.stats.seconds
            before_failures = pipeline.llm_refiner.stats.failures

            started = time.perf_counter()
            predictions, _ = pipeline.process(text)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            predictions = filter_types(predictions, target_types)

            llm_calls = pipeline.llm_refiner.stats.calls - before_calls
            llm_seconds = pipeline.llm_refiner.stats.seconds - before_seconds
            llm_failures = pipeline.llm_refiner.stats.failures - before_failures

            row = {
                "method": "proposed",
                "id": record.get("id"),
                "latency_ms": elapsed_ms,
                "llm_calls": llm_calls,
                "llm_seconds": llm_seconds,
                "llm_failures": llm_failures,
                "ginza_available": True,
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
                f"llm_seconds={llm_seconds:.2f} failures={llm_failures}",
                flush=True,
            )

    print(
        f"[done] completed={len(completed)}/{len(records)} "
        f"processed_this_run={processed_this_run} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
