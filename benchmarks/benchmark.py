from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import yaml

from benchmarks.evaluator import aggregate, bootstrap_f1_ci, score_document
from src.pipeline import PiiPipeline
from src.schemas import PIIEntity, PIIType, PipelineConfig


def load_dataset(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_entities(raw_entities: list[dict]) -> list[PIIEntity]:
    output: list[PIIEntity] = []
    for item in raw_entities:
        try:
            pii_type = PIIType(str(item["type"]))
        except ValueError:
            continue
        output.append(
            PIIEntity(
                type=pii_type,
                text=str(item["text"]),
                start=int(item["start"]),
                end=int(item["end"]),
                score=1.0,
                source="ground_truth",
            )
        )
    return output


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def evaluate_method(
    name: str,
    method_cfg: dict,
    records: list[dict],
    llm_model: str,
    bootstrap_cfg: dict,
) -> tuple[dict, list[dict]]:
    config = PipelineConfig(
        use_regex=bool(method_cfg.get("use_regex", True)),
        use_dict=bool(method_cfg.get("use_dict", True)),
        use_nlp=bool(method_cfg.get("use_nlp", True)),
        use_llm=bool(method_cfg.get("use_llm", True)),
        llm_model=llm_model,
    )
    pipeline = PiiPipeline(config)
    pipeline.reset_stats()

    latencies_ms: list[float] = []
    exact_counts = []
    relaxed_counts = []
    per_document: list[dict] = []

    for record in records:
        text = str(record["text"])
        truths = parse_entities(record.get("entities", []))

        started = time.perf_counter()
        predictions, _ = pipeline.process(text)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(elapsed_ms)

        exact = score_document(truths, predictions, mode="exact")
        relaxed = score_document(truths, predictions, mode="relaxed")
        exact_counts.append(exact)
        relaxed_counts.append(relaxed)

        per_document.append(
            {
                "method": name,
                "id": record.get("id"),
                "latency_ms": elapsed_ms,
                "exact": {"tp": exact.tp, "fp": exact.fp, "fn": exact.fn},
                "relaxed": {"tp": relaxed.tp, "fp": relaxed.fp, "fn": relaxed.fn},
                "predictions": [p.to_dict() for p in predictions],
            }
        )

    exact_total = aggregate(exact_counts)
    relaxed_total = aggregate(relaxed_counts)
    exact_ci = bootstrap_f1_ci(
        exact_counts,
        iterations=int(bootstrap_cfg.get("iterations", 2000)),
        seed=int(bootstrap_cfg.get("seed", 42)),
    )
    relaxed_ci = bootstrap_f1_ci(
        relaxed_counts,
        iterations=int(bootstrap_cfg.get("iterations", 2000)),
        seed=int(bootstrap_cfg.get("seed", 42)),
    )

    llm_failures = pipeline.llm_refiner.stats.failures if pipeline.llm_refiner else 0
    summary = {
        "method": name,
        "documents": len(records),
        "exact_tp": exact_total.tp,
        "exact_fp": exact_total.fp,
        "exact_fn": exact_total.fn,
        "exact_precision": exact_total.precision,
        "exact_recall": exact_total.recall,
        "exact_f1": exact_total.f1,
        "exact_f1_ci95_low": exact_ci[0],
        "exact_f1_ci95_high": exact_ci[1],
        "relaxed_tp": relaxed_total.tp,
        "relaxed_fp": relaxed_total.fp,
        "relaxed_fn": relaxed_total.fn,
        "relaxed_precision": relaxed_total.precision,
        "relaxed_recall": relaxed_total.recall,
        "relaxed_f1": relaxed_total.f1,
        "relaxed_f1_ci95_low": relaxed_ci[0],
        "relaxed_f1_ci95_high": relaxed_ci[1],
        "mean_ms": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
        "p50_ms": percentile(latencies_ms, 0.50),
        "p95_ms": percentile(latencies_ms, 0.95),
        "total_seconds": sum(latencies_ms) / 1000.0,
        "llm_calls": pipeline.stats.llm_calls,
        "llm_call_rate": pipeline.stats.llm_call_rate,
        "llm_seconds": pipeline.stats.llm_seconds,
        "llm_failures": llm_failures,
        "ginza_available": pipeline.ginza_available,
    }
    return summary, per_document


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CSS 2026 PII ablation benchmark")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_path = Path(cfg["dataset"]["path"])
    records = load_dataset(dataset_path)
    if args.limit is not None:
        records = records[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    all_per_document: list[dict] = []
    for method_name, method_cfg in cfg["methods"].items():
        print(f"[benchmark] method={method_name} documents={len(records)}", flush=True)
        summary, per_document = evaluate_method(
            method_name,
            method_cfg,
            records,
            str(cfg["llm"]["model"]),
            cfg.get("bootstrap", {}),
        )
        summaries.append(summary)
        all_per_document.extend(per_document)
        print(
            f"  exact_f1={summary['exact_f1']:.4f} "
            f"relaxed_f1={summary['relaxed_f1']:.4f} "
            f"mean_ms={summary['mean_ms']:.2f} "
            f"llm_call_rate={summary['llm_call_rate']:.3f}",
            flush=True,
        )

    write_csv(output_dir / "summary.csv", summaries)
    with (output_dir / "per_document.jsonl").open("w", encoding="utf-8") as f:
        for row in all_per_document:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "config": cfg,
        "dataset_records_used": len(records),
        "generated_at_unix": time.time(),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[done] {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
