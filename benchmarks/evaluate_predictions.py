from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

from benchmarks.benchmark import filter_types, load_dataset, parse_entities, parse_target_types
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
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", default="results/external_summary.csv")
    parser.add_argument("--bootstrap-iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_path = Path(args.dataset or cfg["dataset"]["path"])
    target_types = parse_target_types(cfg.get("evaluation", {}).get("target_types"))
    iterations = args.bootstrap_iterations or int(cfg.get("bootstrap", {}).get("iterations", 2000))
    seed = args.seed if args.seed is not None else int(cfg.get("bootstrap", {}).get("seed", 42))

    dataset = load_dataset(dataset_path)
    prediction_rows = _load_prediction_rows(Path(args.predictions))

    exact_docs: list[Counts] = []
    relaxed_docs: list[Counts] = []
    per_type_exact: dict[PIIType, Counts] = defaultdict(Counts)
    per_type_relaxed: dict[PIIType, Counts] = defaultdict(Counts)
    latencies: list[float] = []

    for record in dataset:
        record_id = str(record["id"])
        truths = filter_types(parse_entities(record.get("entities", [])), target_types)
        row = prediction_rows.get(record_id, {"predictions": []})
        preds = filter_types(_pred_entities(row.get("predictions", [])), target_types)
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
    exact_ci = bootstrap_f1_ci(exact_docs, iterations=iterations, seed=seed)
    relaxed_ci = bootstrap_f1_ci(relaxed_docs, iterations=iterations, seed=seed)

    summary = {
        "method": args.method,
        "documents": len(dataset),
        "target_types": "+".join(sorted(t.value for t in target_types)) if target_types else "ALL",
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
