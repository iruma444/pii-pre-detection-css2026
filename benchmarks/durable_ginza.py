from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import yaml

from benchmarks.benchmark import filter_types, load_dataset, parse_entities, parse_target_types
from src.pipeline import PiiPipeline
from src.schemas import PipelineConfig


def _repair_and_load_completed(path: Path) -> set[str]:
    """Keep only valid unique JSONL rows and return their document IDs.

    A sudden reboot can leave the final JSONL line partially written.  Rewriting
    only valid unique rows makes the file safe to resume without duplicating
    already completed documents.
    """
    if not path.exists():
        return set()

    valid_rows: list[dict] = []
    completed: set[str] = set()
    changed = False

    with path.open("r", encoding="utf-8", errors="strict") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                changed = True
                continue
            try:
                row = json.loads(line)
                record_id = str(row["id"])
            except Exception:
                print(f"[repair] dropping malformed line {line_number}", flush=True)
                changed = True
                continue
            if record_id in completed:
                print(f"[repair] dropping duplicate id={record_id}", flush=True)
                changed = True
                continue
            valid_rows.append(row)
            completed.add(record_id)

    if changed:
        tmp = path.with_suffix(path.suffix + ".repair")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for row in valid_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    return completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Durable, resumable, low-duty-cycle regex+dict+GiNZA runner"
    )
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--output", default="results/ginza_ultrasafe_500/chunks/live/per_document.jsonl")
    parser.add_argument("--pause-seconds", type=float, default=5.0)
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
        f"[durable-ginza] total={len(records)} completed={len(completed)} "
        f"remaining={remaining} pause={args.pause_seconds}s",
        flush=True,
    )

    pipeline = PiiPipeline(
        PipelineConfig(
            use_regex=True,
            use_dict=True,
            use_nlp=True,
            use_llm=False,
            llm_model=str(cfg["llm"]["model"]),
        )
    )
    if pipeline.ginza_available is not True:
        raise SystemExit("ja_ginza is not available; refusing to record a GiNZA benchmark")

    processed_this_run = 0
    with output.open("a", encoding="utf-8", newline="\n", buffering=1) as f:
        for index, record in enumerate(records, 1):
            record_id = str(record["id"])
            if record_id in completed:
                continue

            text = str(record["text"])
            truths = filter_types(parse_entities(record.get("entities", [])), target_types)

            started = time.perf_counter()
            predictions, _ = pipeline.process(text)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            predictions = filter_types(predictions, target_types)

            row = {
                "method": "regex_dict_ginza",
                "id": record.get("id"),
                "latency_ms": elapsed_ms,
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
                f"latency_ms={elapsed_ms:.2f}",
                flush=True,
            )

            # Deliberately keep the duty cycle low.  The machine has shown WHEA
            # corrected-machine-check events under sustained CPU load, so the
            # research result is checkpointed before every cooldown.
            if args.pause_seconds > 0:
                time.sleep(args.pause_seconds)

    print(
        f"[done] completed={len(completed)}/{len(records)} "
        f"processed_this_run={processed_this_run} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
