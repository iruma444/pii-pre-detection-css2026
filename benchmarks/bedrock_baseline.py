from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from benchmarks.benchmark import load_dataset


AWS_TYPE_MAP = {
    "NAME": "PERSON",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "ADDRESS": "ADDRESS",
    "AGE": "AGE",
    "DRIVER_ID": "DRIVER_ID",
    "CREDIT_DEBIT_CARD_NUMBER": "CREDIT_CARD",
}


def locate_matches(text: str, detected: list[dict]) -> list[dict]:
    """Convert Bedrock match strings to spans without inventing offsets."""
    predictions: list[dict] = []
    used: set[tuple[int, int, str]] = set()

    for item in detected:
        aws_type = str(item.get("type", ""))
        pii_type = AWS_TYPE_MAP.get(aws_type)
        value = str(item.get("match", ""))
        if pii_type is None or not value:
            continue

        start = 0
        chosen = None
        while True:
            pos = text.find(value, start)
            if pos < 0:
                break
            key = (pos, pos + len(value), pii_type)
            if key not in used:
                chosen = key
                break
            start = pos + 1

        if chosen is None:
            continue
        used.add(chosen)
        predictions.append(
            {
                "type": pii_type,
                "text": value,
                "start": chosen[0],
                "end": chosen[1],
                "source": f"bedrock:{aws_type}",
                "action": item.get("action"),
                "detected": item.get("detected", True),
            }
        )

    return sorted(predictions, key=lambda x: x["start"])


def load_completed_ids(path: Path) -> set[str]:
    """Return IDs already written to a JSONL result file for safe resume."""
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Cannot resume: malformed JSONL at {path}:{line_no}: {exc}"
                ) from exc
            if "id" not in row:
                raise SystemExit(f"Cannot resume: missing id at {path}:{line_no}")
            record_id = str(row["id"])
            if record_id in completed:
                raise SystemExit(f"Cannot resume: duplicate id {record_id!r} in {path}")
            completed.add(record_id)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bedrock Guardrails on the exact held-out benchmark")
    parser.add_argument("--dataset", default="datasets/ai4privacy_ja_500.jsonl")
    parser.add_argument("--output", default="results/bedrock_predictions.jsonl")
    parser.add_argument("--guardrail-id", required=True)
    parser.add_argument("--guardrail-version", default="DRAFT")
    parser.add_argument("--region", default="ap-southeast-2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing JSONL file and skip IDs already completed.",
    )
    args = parser.parse_args()

    try:
        import boto3
    except ImportError as exc:
        raise SystemExit("Install boto3 to run the Bedrock baseline: pip install boto3") from exc

    records = load_dataset(Path(args.dataset))
    if args.limit is not None:
        records = records[: args.limit]

    client = boto3.client("bedrock-runtime", region_name=args.region)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = load_completed_ids(output) if args.resume else set()
    remaining = [record for record in records if str(record.get("id")) not in completed_ids]
    if args.resume:
        print(
            f"[resume] completed={len(completed_ids)} remaining={len(remaining)} total={len(records)}",
            flush=True,
        )

    mode = "a" if args.resume else "w"
    with output.open(mode, encoding="utf-8") as f:
        for index, record in enumerate(remaining, 1):
            text = str(record["text"])
            started = time.perf_counter()
            try:
                response = client.apply_guardrail(
                    guardrailIdentifier=args.guardrail_id,
                    guardrailVersion=args.guardrail_version,
                    source="INPUT",
                    content=[{"text": {"text": text}}],
                    outputScope="FULL",
                )
            except Exception:
                print(
                    f"[error] completed_this_run={index - 1}; saved output can be resumed with --resume",
                    flush=True,
                )
                raise
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            pii_entities: list[dict] = []
            for assessment in response.get("assessments", []):
                policy = assessment.get("sensitiveInformationPolicy", {})
                for entity in policy.get("piiEntities", []):
                    if entity.get("detected", True) is False:
                        continue
                    pii_entities.append(entity)

            row = {
                "id": record.get("id"),
                "latency_ms": elapsed_ms,
                "predictions": locate_matches(text, pii_entities),
                "bedrock_action": response.get("action"),
                "usage": response.get("usage"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            completed_total = len(completed_ids) + index
            print(f"[{completed_total}/{len(records)}] {record.get('id')}", flush=True)

    print(f"[done] {output}")


if __name__ == "__main__":
    main()
