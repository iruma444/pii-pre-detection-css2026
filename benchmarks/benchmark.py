from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import yaml

from benchmarks.environment import collect_environment
from benchmarks.evaluator import Counts, aggregate, bootstrap_f1_ci, score_document
from src.extractors.llm_only_extractor import FullTextLlmExtractor
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


def parse_target_types(raw_types: list[str] | None) -> set[PIIType] | None:
    if not raw_types:
        return None
    return {PIIType(value) for value in raw_types}


def filter_types(entities: list[PIIEntity], target_types: set[PIIType] | None) -> list[PIIEntity]:
    if target_types is None:
        return entities
    return [entity for entity in entities if entity.type in target_types]


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


def _type_counts(
    truths: list[PIIEntity], predictions: list[PIIEntity], mode: str
) -> dict[PIIType, Counts]:
    result: dict[PIIType, Counts] = {}
    present_types = {e.type for e in truths} | {e.type for e in predictions}
    for pii_type in present_types:
        result[pii_type] = score_document(
            [e for e in truths if e.type == pii_type],
            [e for e in predictions if e.type == pii_type],
            mode=mode,
        )
    return result


def evaluate_method(
    name: str,
    method_cfg: dict,
    records: list[dict],
    llm_model: str,
    bootstrap_cfg: dict,
    target_types: set[PIIType] | None,
) -> tuple[dict, list[dict], list[dict]]:
    mode = str(method_cfg.get("mode", "pipeline"))
    pipeline: PiiPipeline | None = None
    llm_only: FullTextLlmExtractor | None = None

    if mode == "llm_only":
        llm_only = FullTextLlmExtractor(llm_model)
    else:
        config = PipelineConfig(
            use_regex=bool(method_cfg.get("use_regex", True)),
            use_dict=bool(method_cfg.get("use_dict", True)),
            use_nlp=bool(method_cfg.get("use_nlp", True)),
            use_llm=bool(method_cfg.get("use_llm", True)),
            use_boundary_split=bool(method_cfg.get("use_boundary_split", True)),
            llm_model=llm_model,
        )
        pipeline = PiiPipeline(config)
        pipeline.reset_stats()
        if config.use_nlp and pipeline.ginza_available is not True:
            raise SystemExit(
                f"Method {name!r} requires ja_ginza, but it is not available in this Python environment. "
                "Activate the benchmark virtual environment before running this method."
            )

    latencies_ms: list[float] = []
    exact_counts: list[Counts] = []
    relaxed_counts: list[Counts] = []
    exact_type_totals: dict[PIIType, Counts] = defaultdict(Counts)
    relaxed_type_totals: dict[PIIType, Counts] = defaultdict(Counts)
    per_document: list[dict] = []

    for record in records:
        text = str(record["text"])
        truths = filter_types(parse_entities(record.get("entities", [])), target_types)

        started = time.perf_counter()
        if llm_only is not None:
            predictions = llm_only.extract(text)
        else:
            assert pipeline is not None
            predictions, _ = pipeline.process(text)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        predictions = filter_types(predictions, target_types)
        latencies_ms.append(elapsed_ms)

        exact = score_document(truths, predictions, mode="exact")
        relaxed = score_document(truths, predictions, mode="relaxed")
        exact_counts.append(exact)
        relaxed_counts.append(relaxed)

        for pii_type, counts in _type_counts(truths, predictions, "exact").items():
            exact_type_totals[pii_type] = exact_type_totals[pii_type] + counts
        for pii_type, counts in _type_counts(truths, predictions, "relaxed").items():
            relaxed_type_totals[pii_type] = relaxed_type_totals[pii_type] + counts

        per_document.append(
            {
                "method": name,
                "id": record.get("id"),
                "latency_ms": elapsed_ms,
                "exact": {"tp": exact.tp, "fp": exact.fp, "fn": exact.fn},
                "relaxed": {"tp": relaxed.tp, "fp": relaxed.fp, "fn": relaxed.fn},
                "truth": [t.to_dict() for t in truths],
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

    if llm_only is not None:
        llm_calls = llm_only.stats.calls
        llm_seconds = llm_only.stats.seconds
        llm_failures = llm_only.stats.failures
        llm_call_rate = llm_calls / len(records) if records else 0.0
        ginza_available = None
    else:
        assert pipeline is not None
        llm_calls = pipeline.stats.llm_calls
        llm_seconds = pipeline.stats.llm_seconds
        llm_failures = pipeline.llm_refiner.stats.failures if pipeline.llm_refiner else 0
        llm_call_rate = pipeline.stats.llm_call_rate
        ginza_available = pipeline.ginza_available

    summary = {
        "method": name,
        "documents": len(records),
        "target_types": "+".join(sorted(t.value for t in target_types)) if target_types else "ALL",
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
        "llm_calls": llm_calls,
        "llm_call_rate": llm_call_rate,
        "llm_seconds": llm_seconds,
        "llm_failures": llm_failures,
        "ginza_available": ginza_available,
    }

    per_type_rows: list[dict] = []
    all_types = sorted(
        set(exact_type_totals) | set(relaxed_type_totals), key=lambda t: t.value
    )
    for pii_type in all_types:
        exact_t = exact_type_totals.get(pii_type, Counts())
        relaxed_t = relaxed_type_totals.get(pii_type, Counts())
        per_type_rows.append(
            {
                "method": name,
                "type": pii_type.value,
                "exact_tp": exact_t.tp,
                "exact_fp": exact_t.fp,
                "exact_fn": exact_t.fn,
                "exact_precision": exact_t.precision,
                "exact_recall": exact_t.recall,
                "exact_f1": exact_t.f1,
                "relaxed_tp": relaxed_t.tp,
                "relaxed_fp": relaxed_t.fp,
                "relaxed_fn": relaxed_t.fn,
                "relaxed_precision": relaxed_t.precision,
                "relaxed_recall": relaxed_t.recall,
                "relaxed_f1": relaxed_t.f1,
            }
        )

    return summary, per_document, per_type_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _method_uses_llm(method_cfg: dict) -> bool:
    return str(method_cfg.get("mode", "pipeline")) == "llm_only" or bool(
        method_cfg.get("use_llm", True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CSS 2026 PII ablation benchmark")
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--method", action="append", help="Run only selected method(s)")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_path = Path(cfg["dataset"]["path"])
    records = load_dataset(dataset_path)
    if args.limit is not None:
        records = records[: args.limit]

    target_types = parse_target_types(cfg.get("evaluation", {}).get("target_types"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested = set(args.method or [])
    selected_methods = [
        (name, method_cfg)
        for name, method_cfg in cfg["methods"].items()
        if not requested or name in requested
    ]
    include_ollama = any(_method_uses_llm(method_cfg) for _, method_cfg in selected_methods)

    summaries: list[dict] = []
    all_per_document: list[dict] = []
    all_per_type: list[dict] = []
    for method_name, method_cfg in selected_methods:
        print(f"[benchmark] method={method_name} documents={len(records)}", flush=True)
        summary, per_document, per_type = evaluate_method(
            method_name,
            method_cfg,
            records,
            str(cfg["llm"]["model"]),
            cfg.get("bootstrap", {}),
            target_types,
        )
        summaries.append(summary)
        all_per_document.extend(per_document)
        all_per_type.extend(per_type)
        print(
            f"  exact_f1={summary['exact_f1']:.4f} "
            f"relaxed_f1={summary['relaxed_f1']:.4f} "
            f"mean_ms={summary['mean_ms']:.2f} "
            f"llm_call_rate={summary['llm_call_rate']:.3f}",
            flush=True,
        )

    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "per_type.csv", all_per_type)
    with (output_dir / "per_document.jsonl").open("w", encoding="utf-8") as f:
        for row in all_per_document:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    environment = collect_environment(
        str(cfg["llm"]["model"]), include_ollama=include_ollama
    )
    environment.update(
        {
            "config": cfg,
            "dataset_records_used": len(records),
            "generated_at_unix": time.time(),
            "ollama_probed": include_ollama,
        }
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[done] {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
