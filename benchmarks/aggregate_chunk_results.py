from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from benchmarks.benchmark import parse_target_types, percentile
from benchmarks.evaluator import Counts, aggregate, bootstrap_f1_ci, score_document
from src.schemas import PIIEntity, PIIType


def _entity(raw: dict) -> PIIEntity:
    return PIIEntity(
        type=PIIType(str(raw["type"])),
        text=str(raw["text"]),
        start=int(raw["start"]),
        end=int(raw["end"]),
        score=float(raw.get("score", 1.0)),
        source=str(raw.get("source", "chunk")),
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate independently executed benchmark chunks")
    parser.add_argument("--chunks-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--expected-documents", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    target_types = parse_target_types(cfg.get("evaluation", {}).get("target_types"))
    bootstrap_cfg = cfg.get("bootstrap", {})
    iterations = int(bootstrap_cfg.get("iterations", 2000))
    seed = int(bootstrap_cfg.get("seed", 42))

    chunk_files = sorted(Path(args.chunks_dir).glob("*/per_document.jsonl"))
    if not chunk_files:
        raise SystemExit(f"No chunk per_document.jsonl files found under {args.chunks_dir}")

    rows_by_id: dict[str, dict] = {}
    for path in chunk_files:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = str(row["id"])
                if record_id in rows_by_id:
                    raise SystemExit(f"Duplicate document id across chunks: {record_id}")
                rows_by_id[record_id] = row

    rows = list(rows_by_id.values())
    if args.expected_documents is not None and len(rows) != args.expected_documents:
        raise SystemExit(
            f"Expected {args.expected_documents} documents but found {len(rows)}. "
            "Resume the chunk runner before aggregating."
        )

    exact_docs: list[Counts] = []
    relaxed_docs: list[Counts] = []
    per_type_exact: dict[PIIType, Counts] = defaultdict(Counts)
    per_type_relaxed: dict[PIIType, Counts] = defaultdict(Counts)
    latencies: list[float] = []

    for row in rows:
        truths = [_entity(x) for x in row.get("truth", [])]
        preds = [_entity(x) for x in row.get("predictions", [])]

        exact = score_document(truths, preds, mode="exact")
        relaxed = score_document(truths, preds, mode="relaxed")
        exact_docs.append(exact)
        relaxed_docs.append(relaxed)
        latencies.append(float(row.get("latency_ms", 0.0)))

        all_types = {x.type for x in truths} | {x.type for x in preds}
        for pii_type in all_types:
            t = [x for x in truths if x.type == pii_type]
            p = [x for x in preds if x.type == pii_type]
            per_type_exact[pii_type] = per_type_exact[pii_type] + score_document(t, p, mode="exact")
            per_type_relaxed[pii_type] = per_type_relaxed[pii_type] + score_document(t, p, mode="relaxed")

    exact = aggregate(exact_docs)
    relaxed = aggregate(relaxed_docs)
    exact_ci = bootstrap_f1_ci(exact_docs, iterations=iterations, seed=seed)
    relaxed_ci = bootstrap_f1_ci(relaxed_docs, iterations=iterations, seed=seed)

    method = str(rows[0].get("method", "chunked"))
    target_label = "+".join(sorted(t.value for t in target_types)) if target_types else "ALL"
    summary = {
        "method": method,
        "documents": len(rows),
        "target_types": target_label,
        "exact_tp": exact.tp,
        "exact_fp": exact.fp,
        "exact_fn": exact.fn,
        "exact_precision": exact.precision,
        "exact_recall": exact.recall,
        "exact_f1": exact.f1,
        "exact_f1_ci95_low": exact_ci[0],
        "exact_f1_ci95_high": exact_ci[1],
        "relaxed_tp": relaxed.tp,
        "relaxed_fp": relaxed.fp,
        "relaxed_fn": relaxed.fn,
        "relaxed_precision": relaxed.precision,
        "relaxed_recall": relaxed.recall,
        "relaxed_f1": relaxed.f1,
        "relaxed_f1_ci95_low": relaxed_ci[0],
        "relaxed_f1_ci95_high": relaxed_ci[1],
        "mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "total_seconds": sum(latencies) / 1000.0,
        "llm_calls": 0,
        "llm_call_rate": 0.0,
        "llm_seconds": 0.0,
        "llm_failures": 0,
        "ginza_available": True if method == "regex_dict_ginza" else "",
    }

    per_type_rows: list[dict] = []
    for pii_type in sorted(set(per_type_exact) | set(per_type_relaxed), key=lambda x: x.value):
        e = per_type_exact.get(pii_type, Counts())
        r = per_type_relaxed.get(pii_type, Counts())
        per_type_rows.append(
            {
                "method": method,
                "type": pii_type.value,
                "exact_tp": e.tp,
                "exact_fp": e.fp,
                "exact_fn": e.fn,
                "exact_precision": e.precision,
                "exact_recall": e.recall,
                "exact_f1": e.f1,
                "relaxed_tp": r.tp,
                "relaxed_fp": r.fp,
                "relaxed_fn": r.fn,
                "relaxed_precision": r.precision,
                "relaxed_recall": r.recall,
                "relaxed_f1": r.f1,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", [summary])
    _write_csv(output_dir / "per_type.csv", per_type_rows)
    with (output_dir / "per_document.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] {output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
