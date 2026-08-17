from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import yaml

from benchmarks.benchmark import parse_target_types
from benchmarks.evaluator import Counts, aggregate, score_document
from src.schemas import PIIEntity, PIIType


def _entity(raw: dict) -> PIIEntity:
    return PIIEntity(
        type=PIIType(str(raw["type"])),
        text=str(raw["text"]),
        start=int(raw["start"]),
        end=int(raw["end"]),
        score=float(raw.get("score", 1.0)),
        source=str(raw.get("source", "analysis")),
    )


def _load_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row["id"])
            if record_id in rows:
                raise SystemExit(f"Duplicate document id in {path}: {record_id}")
            rows[record_id] = row
    return rows


def _truth_signature(row: dict) -> list[tuple[str, int, int, str]]:
    return sorted(
        (
            str(x["type"]),
            int(x["start"]),
            int(x["end"]),
            str(x["text"]),
        )
        for x in row.get("truth", [])
    )


def _counts_for_row(
    row: dict,
    *,
    mode: str,
    target_type: PIIType | None,
    target_types: set[PIIType] | None,
) -> Counts:
    truths = [_entity(x) for x in row.get("truth", [])]
    preds = [_entity(x) for x in row.get("predictions", [])]

    if target_types is not None:
        truths = [x for x in truths if x.type in target_types]
        preds = [x for x in preds if x.type in target_types]
    if target_type is not None:
        truths = [x for x in truths if x.type == target_type]
        preds = [x for x in preds if x.type == target_type]

    return score_document(truths, preds, mode=mode)


def _paired_delta_ci(
    baseline_docs: list[Counts],
    candidate_docs: list[Counts],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float, float]:
    if len(baseline_docs) != len(candidate_docs):
        raise ValueError("Paired samples must have the same number of documents")
    if not baseline_docs:
        return 0.0, 0.0, 0.0, 0.0

    observed = aggregate(candidate_docs).f1 - aggregate(baseline_docs).f1
    rng = random.Random(seed)
    n = len(baseline_docs)
    deltas: list[float] = []

    for _ in range(iterations):
        indices = [rng.randrange(n) for _ in range(n)]
        baseline = aggregate(baseline_docs[i] for i in indices)
        candidate = aggregate(candidate_docs[i] for i in indices)
        deltas.append(candidate.f1 - baseline.f1)

    deltas.sort()
    low_index = max(0, int(0.025 * iterations))
    high_index = min(iterations - 1, int(0.975 * iterations) - 1)
    probability_positive = sum(delta > 0.0 for delta in deltas) / iterations
    return observed, deltas[low_index], deltas[high_index], probability_positive


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired document-bootstrap F1 difference between two saved benchmark runs"
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default="results/paired_bootstrap.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    target_types = parse_target_types(cfg.get("evaluation", {}).get("target_types"))
    iterations = args.iterations or int(cfg.get("bootstrap", {}).get("iterations", 2000))
    seed = args.seed if args.seed is not None else int(cfg.get("bootstrap", {}).get("seed", 42))
    if iterations <= 0:
        raise SystemExit("--iterations must be positive")

    baseline_rows = _load_rows(Path(args.baseline))
    candidate_rows = _load_rows(Path(args.candidate))

    baseline_ids = set(baseline_rows)
    candidate_ids = set(candidate_rows)
    if baseline_ids != candidate_ids:
        missing_candidate = sorted(baseline_ids - candidate_ids)[:5]
        missing_baseline = sorted(candidate_ids - baseline_ids)[:5]
        raise SystemExit(
            "Paired bootstrap requires identical document ids. "
            f"missing_in_candidate={missing_candidate} missing_in_baseline={missing_baseline}"
        )

    ids = sorted(baseline_ids)
    for record_id in ids:
        if _truth_signature(baseline_rows[record_id]) != _truth_signature(candidate_rows[record_id]):
            raise SystemExit(f"Ground-truth mismatch for document id {record_id}")

    scopes: list[tuple[str, PIIType | None]] = [("ALL", None)]
    if target_types is not None:
        scopes.extend((pii_type.value, pii_type) for pii_type in sorted(target_types, key=lambda x: x.value))

    output_rows: list[dict] = []
    for scope_name, target_type in scopes:
        for mode in ("exact", "relaxed"):
            baseline_docs = [
                _counts_for_row(
                    baseline_rows[record_id],
                    mode=mode,
                    target_type=target_type,
                    target_types=target_types,
                )
                for record_id in ids
            ]
            candidate_docs = [
                _counts_for_row(
                    candidate_rows[record_id],
                    mode=mode,
                    target_type=target_type,
                    target_types=target_types,
                )
                for record_id in ids
            ]

            baseline_f1 = aggregate(baseline_docs).f1
            candidate_f1 = aggregate(candidate_docs).f1
            delta, ci_low, ci_high, probability_positive = _paired_delta_ci(
                baseline_docs,
                candidate_docs,
                iterations=iterations,
                seed=seed,
            )
            output_rows.append(
                {
                    "scope": scope_name,
                    "mode": mode,
                    "documents": len(ids),
                    "baseline_f1": baseline_f1,
                    "candidate_f1": candidate_f1,
                    "delta_f1": delta,
                    "delta_ci95_low": ci_low,
                    "delta_ci95_high": ci_high,
                    "bootstrap_probability_delta_gt_0": probability_positive,
                    "iterations": iterations,
                    "seed": seed,
                }
            )

    output = Path(args.output)
    _write_csv(output, output_rows)
    print(json.dumps(output_rows, ensure_ascii=False, indent=2))
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
