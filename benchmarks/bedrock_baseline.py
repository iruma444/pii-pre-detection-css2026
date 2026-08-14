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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Bedrock Guardrails on the exact held-out benchmark")
    parser.add_argument("--dataset", default="datasets/ai4privacy_ja_500.jsonl")
    parser.add_argument("--output", default="results/bedrock_predictions.jsonl")
    parser.add_argument("--guardrail-id", required=True)
    parser.add_argument("--guardrail-version", default="DRAFT")
    parser.add_argument("--region", default="ap-southeast-2")
    parser.add_argument("--limit", type=int, default=None)
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

    with output.open("w", encoding="utf-8") as f:
        for index, record in enumerate(records, 1):
            text = str(record["text"])
            started = time.perf_counter()
            response = client.apply_guardrail(
                guardrailIdentifier=args.guardrail_id,
                guardrailVersion=args.guardrail_version,
                source="INPUT",
                content=[{"text": {"text": text}}],
                outputScope="FULL",
            )
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
            print(f"[{index}/{len(records)}] {record.get('id')}", flush=True)

    print(f"[done] {output}")


if __name__ == "__main__":
    main()
