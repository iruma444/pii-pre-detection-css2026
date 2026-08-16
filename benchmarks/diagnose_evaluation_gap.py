from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

from benchmarks.benchmark import (
    filter_types,
    load_dataset,
    parse_entities,
    parse_target_types,
)
from benchmarks.evaluator import Counts, aggregate, exact_match, relaxed_match, score_document
from src.extractors.regex_extractor import RegexExtractor
from src.schemas import PIIEntity, PIIType


def _pred_entities(raw: list[dict]) -> list[PIIEntity]:
    output: list[PIIEntity] = []
    for item in raw:
        try:
            pii_type = PIIType(str(item["type"]))
        except (KeyError, ValueError):
            continue
        output.append(
            PIIEntity(
                type=pii_type,
                text=str(item.get("text", "")),
                start=int(item["start"]),
                end=int(item["end"]),
                score=float(item.get("score", 1.0)),
                source=str(item.get("source", "diagnostic")),
            )
        )
    return output


def _load_result_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row:
                raise SystemExit(f"Missing id in {path}:{line_no}")
            record_id = str(row["id"])
            if record_id in rows:
                raise SystemExit(f"Duplicate id {record_id!r} in {path}")
            rows[record_id] = row
    return rows


def _entity_key(entity: PIIEntity) -> tuple[str, int, int, str]:
    return (entity.type.value, entity.start, entity.end, entity.text)


def _presence_counts(
    truths: list[PIIEntity],
    predictions: list[PIIEntity],
    target_types: set[PIIType],
) -> tuple[Counts, dict[PIIType, Counts]]:
    """Score only whether a PII type is present in a document.

    This deliberately ignores entity multiplicity and span boundaries.  It is a
    diagnostic proxy for earlier type-level/document-level reporting, not a
    replacement for the stricter entity-span benchmark.
    """
    total = Counts()
    per_type: dict[PIIType, Counts] = {}
    truth_types = {x.type for x in truths}
    pred_types = {x.type for x in predictions}

    for pii_type in target_types:
        truth_present = pii_type in truth_types
        pred_present = pii_type in pred_types
        counts = Counts(
            tp=int(truth_present and pred_present),
            fp=int((not truth_present) and pred_present),
            fn=int(truth_present and (not pred_present)),
        )
        per_type[pii_type] = counts
        total = total + counts
    return total, per_type


def _working_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return normalized if len(normalized) == len(text) else text


def _truth_has_match(
    truth: PIIEntity,
    predictions: list[PIIEntity],
    *,
    mode: str,
) -> bool:
    matcher = exact_match if mode == "exact" else relaxed_match
    return any(matcher(truth, pred) for pred in predictions)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _method_name(path: Path, rows: dict[str, dict]) -> str:
    for row in rows.values():
        method = str(row.get("method", "")).strip()
        if method:
            return method
    parent = path.parent.name
    return parent or path.stem


def _format_counts(counts: Counts) -> str:
    return (
        f"P={counts.precision:.3f} R={counts.recall:.3f} "
        f"F1={counts.f1:.3f} (TP={counts.tp} FP={counts.fp} FN={counts.fn})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose why strict entity-span scores differ from earlier type-level results. "
            "No LLM inference is performed."
        )
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="per_document.jsonl or external prediction JSONL; may be repeated",
    )
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default="results/evaluation_gap_diagnostic")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum diagnostic miss examples per method/type/category",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset_path = Path(args.dataset or cfg["dataset"]["path"])
    target_types = parse_target_types(cfg.get("evaluation", {}).get("target_types"))
    if target_types is None:
        target_types = set(PIIType)

    dataset = load_dataset(dataset_path)
    records_by_id = {str(record["id"]): record for record in dataset}
    if len(records_by_id) != len(dataset):
        raise SystemExit("Dataset contains duplicate document ids")

    # Run the raw regex extractor once on the frozen dataset.  This is useful for
    # separating "the regex cannot match the ground truth" from "a later pipeline
    # stage removed or changed a match".
    regex = RegexExtractor()
    regex_by_id: dict[str, list[PIIEntity]] = {}
    regex_exact_docs: list[Counts] = []
    regex_relaxed_docs: list[Counts] = []
    regex_per_type_exact: dict[PIIType, Counts] = defaultdict(Counts)
    regex_per_type_relaxed: dict[PIIType, Counts] = defaultdict(Counts)
    truth_entity_counts: dict[PIIType, int] = defaultdict(int)
    truth_document_counts: dict[PIIType, int] = defaultdict(int)

    for record in dataset:
        record_id = str(record["id"])
        text = str(record["text"])
        truths = filter_types(parse_entities(record.get("entities", [])), target_types)
        raw_regex = filter_types(regex.extract(_working_text(text)), target_types)
        regex_by_id[record_id] = raw_regex
        regex_exact_docs.append(score_document(truths, raw_regex, mode="exact"))
        regex_relaxed_docs.append(score_document(truths, raw_regex, mode="relaxed"))

        for pii_type in target_types:
            t = [x for x in truths if x.type == pii_type]
            p = [x for x in raw_regex if x.type == pii_type]
            truth_entity_counts[pii_type] += len(t)
            truth_document_counts[pii_type] += int(bool(t))
            regex_per_type_exact[pii_type] = regex_per_type_exact[pii_type] + score_document(
                t, p, mode="exact"
            )
            regex_per_type_relaxed[pii_type] = regex_per_type_relaxed[pii_type] + score_document(
                t, p, mode="relaxed"
            )

    regex_summary_rows: list[dict] = []
    for pii_type in sorted(target_types, key=lambda x: x.value):
        exact = regex_per_type_exact.get(pii_type, Counts())
        relaxed = regex_per_type_relaxed.get(pii_type, Counts())
        regex_summary_rows.append(
            {
                "type": pii_type.value,
                "truth_entities": truth_entity_counts[pii_type],
                "truth_documents": truth_document_counts[pii_type],
                "regex_exact_tp": exact.tp,
                "regex_exact_recall": exact.recall,
                "regex_relaxed_tp": relaxed.tp,
                "regex_relaxed_recall": relaxed.recall,
                "regex_predictions": exact.tp + exact.fp,
            }
        )

    summary_rows: list[dict] = []
    per_type_rows: list[dict] = []
    survival_rows: list[dict] = []
    examples: list[dict] = []
    example_counts: dict[tuple[str, str, str], int] = defaultdict(int)

    for result_arg in args.result:
        result_path = Path(result_arg)
        result_rows = _load_result_rows(result_path)
        method = _method_name(result_path, result_rows)

        exact_docs: list[Counts] = []
        relaxed_docs: list[Counts] = []
        presence_docs: list[Counts] = []
        per_type_exact: dict[PIIType, Counts] = defaultdict(Counts)
        per_type_relaxed: dict[PIIType, Counts] = defaultdict(Counts)
        per_type_presence: dict[PIIType, Counts] = defaultdict(Counts)
        pred_entity_counts: dict[PIIType, int] = defaultdict(int)
        pred_document_counts: dict[PIIType, int] = defaultdict(int)
        regex_hit_final_miss: dict[PIIType, int] = defaultdict(int)
        regex_relaxed_hits: dict[PIIType, int] = defaultdict(int)
        final_relaxed_hits: dict[PIIType, int] = defaultdict(int)
        stored_truth_mismatch_documents = 0

        for record in dataset:
            record_id = str(record["id"])
            text = str(record["text"])
            truths = filter_types(parse_entities(record.get("entities", [])), target_types)
            row = result_rows.get(record_id, {"predictions": []})
            preds = filter_types(_pred_entities(row.get("predictions", [])), target_types)
            raw_regex = regex_by_id[record_id]

            if "truth" in row:
                stored_truth = filter_types(_pred_entities(row.get("truth", [])), target_types)
                if sorted(_entity_key(x) for x in stored_truth) != sorted(
                    _entity_key(x) for x in truths
                ):
                    stored_truth_mismatch_documents += 1

            exact = score_document(truths, preds, mode="exact")
            relaxed = score_document(truths, preds, mode="relaxed")
            presence, presence_by_type = _presence_counts(truths, preds, target_types)
            exact_docs.append(exact)
            relaxed_docs.append(relaxed)
            presence_docs.append(presence)

            for pii_type in target_types:
                t = [x for x in truths if x.type == pii_type]
                p = [x for x in preds if x.type == pii_type]
                per_type_exact[pii_type] = per_type_exact[pii_type] + score_document(
                    t, p, mode="exact"
                )
                per_type_relaxed[pii_type] = per_type_relaxed[pii_type] + score_document(
                    t, p, mode="relaxed"
                )
                per_type_presence[pii_type] = (
                    per_type_presence[pii_type] + presence_by_type[pii_type]
                )
                pred_entity_counts[pii_type] += len(p)
                pred_document_counts[pii_type] += int(bool(p))

                regex_type = [x for x in raw_regex if x.type == pii_type]
                for truth in t:
                    regex_hit = _truth_has_match(truth, regex_type, mode="relaxed")
                    final_hit = _truth_has_match(truth, p, mode="relaxed")
                    regex_relaxed_hits[pii_type] += int(regex_hit)
                    final_relaxed_hits[pii_type] += int(final_hit)
                    if regex_hit and not final_hit:
                        regex_hit_final_miss[pii_type] += 1
                        key = (method, pii_type.value, "regex_hit_final_miss")
                        if example_counts[key] < args.max_examples:
                            examples.append(
                                {
                                    "method": method,
                                    "category": "regex_hit_final_miss",
                                    "id": record_id,
                                    "type": pii_type.value,
                                    "truth": truth.to_dict(),
                                    "same_type_final_predictions": [x.to_dict() for x in p],
                                    "same_type_raw_regex": [x.to_dict() for x in regex_type],
                                    "context": text[max(0, truth.start - 40) : min(len(text), truth.end + 40)],
                                }
                            )
                            example_counts[key] += 1
                    elif not final_hit:
                        key = (method, pii_type.value, "final_relaxed_miss")
                        if example_counts[key] < args.max_examples:
                            examples.append(
                                {
                                    "method": method,
                                    "category": "final_relaxed_miss",
                                    "id": record_id,
                                    "type": pii_type.value,
                                    "truth": truth.to_dict(),
                                    "same_type_final_predictions": [x.to_dict() for x in p],
                                    "same_type_raw_regex": [x.to_dict() for x in regex_type],
                                    "context": text[max(0, truth.start - 40) : min(len(text), truth.end + 40)],
                                }
                            )
                            example_counts[key] += 1

        missing_result_documents = len(set(records_by_id) - set(result_rows))
        extra_result_documents = len(set(result_rows) - set(records_by_id))
        exact_total = aggregate(exact_docs)
        relaxed_total = aggregate(relaxed_docs)
        presence_total = aggregate(presence_docs)

        summary_rows.append(
            {
                "method": method,
                "result_path": str(result_path),
                "dataset_documents": len(dataset),
                "result_documents": len(result_rows),
                "missing_result_documents": missing_result_documents,
                "extra_result_documents": extra_result_documents,
                "stored_truth_mismatch_documents": stored_truth_mismatch_documents,
                "truth_entities": sum(truth_entity_counts.values()),
                "truth_document_type_pairs": sum(truth_document_counts.values()),
                "pred_entities": sum(pred_entity_counts.values()),
                "pred_document_type_pairs": sum(pred_document_counts.values()),
                "span_exact_precision": exact_total.precision,
                "span_exact_recall": exact_total.recall,
                "span_exact_f1": exact_total.f1,
                "span_relaxed_precision": relaxed_total.precision,
                "span_relaxed_recall": relaxed_total.recall,
                "span_relaxed_f1": relaxed_total.f1,
                "presence_precision": presence_total.precision,
                "presence_recall": presence_total.recall,
                "presence_f1": presence_total.f1,
            }
        )

        for pii_type in sorted(target_types, key=lambda x: x.value):
            exact = per_type_exact.get(pii_type, Counts())
            relaxed = per_type_relaxed.get(pii_type, Counts())
            presence = per_type_presence.get(pii_type, Counts())
            per_type_rows.append(
                {
                    "method": method,
                    "type": pii_type.value,
                    "truth_entities": truth_entity_counts[pii_type],
                    "truth_documents": truth_document_counts[pii_type],
                    "pred_entities": pred_entity_counts[pii_type],
                    "pred_documents": pred_document_counts[pii_type],
                    "span_exact_precision": exact.precision,
                    "span_exact_recall": exact.recall,
                    "span_exact_f1": exact.f1,
                    "span_relaxed_precision": relaxed.precision,
                    "span_relaxed_recall": relaxed.recall,
                    "span_relaxed_f1": relaxed.f1,
                    "presence_precision": presence.precision,
                    "presence_recall": presence.recall,
                    "presence_f1": presence.f1,
                }
            )
            survival_rows.append(
                {
                    "method": method,
                    "type": pii_type.value,
                    "truth_entities": truth_entity_counts[pii_type],
                    "raw_regex_relaxed_truth_hits": regex_relaxed_hits[pii_type],
                    "final_relaxed_truth_hits": final_relaxed_hits[pii_type],
                    "regex_hit_but_final_miss": regex_hit_final_miss[pii_type],
                }
            )

        print(f"\n=== {method} ===")
        print(f"result: {result_path}")
        print(f"span exact   : {_format_counts(exact_total)}")
        print(f"span relaxed : {_format_counts(relaxed_total)}")
        print(f"type presence: {_format_counts(presence_total)}")
        print(
            "alignment    : "
            f"missing_results={missing_result_documents} "
            f"extra_results={extra_result_documents} "
            f"stored_truth_mismatch_docs={stored_truth_mismatch_documents}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "per_type.csv", per_type_rows)
    _write_csv(output_dir / "regex_coverage.csv", regex_summary_rows)
    _write_csv(output_dir / "regex_survival.csv", survival_rows)
    with (output_dir / "miss_examples.jsonl").open("w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    regex_exact = aggregate(regex_exact_docs)
    regex_relaxed = aggregate(regex_relaxed_docs)
    print("\n=== Raw RegexExtractor on the frozen dataset ===")
    print(f"span exact   : {_format_counts(regex_exact)}")
    print(f"span relaxed : {_format_counts(regex_relaxed)}")
    print("per-type relaxed recall:")
    for row in regex_summary_rows:
        if row["truth_entities"]:
            print(
                f"  {row['type']:<12} truth={row['truth_entities']:<4} "
                f"regex_recall={row['regex_relaxed_recall']:.3f}"
            )

    print(f"\n[done] {output_dir}")
    print("  summary.csv          strict span vs document/type presence")
    print("  per_type.csv         same comparison by PII type")
    print("  regex_coverage.csv   can the raw regex match frozen ground truth?")
    print("  regex_survival.csv   raw-regex hits that disappear in final predictions")
    print("  miss_examples.jsonl  small set of concrete misses for inspection")


if __name__ == "__main__":
    main()
