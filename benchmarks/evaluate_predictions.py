from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from benchmarks.benchmark import load_dataset, parse_entities
from benchmarks.evaluator import Counts, aggregate, bootstrap_f1_ci, score_document
from src.schemas import PIIEntity, PIIType


def _load_prediction_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def _pred_entities(raw: list[dict]) -> list[PIIEntity]:
    output: list[PIIEntity] = []
    for item in raw:
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
                source=str(item.get("source", "external")),
            )
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/ai4privacy_ja_500.jsonl")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", default="results/external_summary.csv")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = load_dataset(Path(args.dataset))
    prediction_rows = _load_prediction_rows(Path(args.predictions))

    exact_docs: list[Counts] = []
    relaxed_docs: list[Counts] = []
    per_type_exact: dict[PIIType, Counts] = defaultdict(Counts)
    per_type_relaxed: dict[PIIType, Counts] = defaultdict(Counts)
    latencies: list[float] = []

    for record in dataset:
        record_id = str(record["id"])
        truths = parse_entities(record.get("entities", []))
        row = prediction_rows.get(record_id, {"predictions": []})
        preds = _pred_entities(row.get("predictions", []))
        if "latency_ms" in row:
            latencies.append(float(row["latency_ms"]))

        exact_docs.append(score_document(truths, preds, mode="exact"))
        relaxed_docs.append(score_document(truths, preds, mode="relaxed"))

        all_types = {x.type for x in truths} | {x.type for x in preds}
        for pii_type in all_types:
            t = [x for x in truths if x.type == pii_type]
            p = [x for x in preds if x.type == pii_type]
            per_type_exact[pii_type] = per_type_exact[pii_type] + score_document(t, p, mode="exact")
            per_type_relaxed[pii_type] = per_type_relaxed[pii_type] + score_document(t, p, mode="relaxed")

    exact = aggregate(exact_docs)
    relaxed = aggregate(relaxed_docs)
    exact_ci = bootstrap_f1_ci(exact_docs, iterations=args.bootstrap_iterations, seed=args.seed)
    relaxed_ci = bootstrap_f1_ci(relaxed_docs, iterations=args.bootstrap_iterations, seed=args.seed)

    summary = {
        "method": args.method,
        "documents": len(dataset),
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
        "mean_ms": sum(latencies) / len(latencies) if latencies else "",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    per_type_output = output.with_name(output.stem + "_per_type.csv")
    rows = []
    for pii_type in sorted(set(per_type_exact) | set(per_type_relaxed), key=lambda x: x.value):
        e = per_type_exact.get(pii_type, Counts())
        r = per_type_relaxed.get(pii_type, Counts())
        rows.append(
            {
                "method": args.method,
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
    with per_type_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["method", "type"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
