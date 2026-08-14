from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


DATASET_NAME = "ai4privacy/pii-masking-openpii-1.5m"

LABEL_MAP = {
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "EMAIL": "EMAIL",
    "TELEPHONENUM": "PHONE",
    "STREET": "ADDRESS",
    "CITY": "ADDRESS",
    "ZIPCODE": "ADDRESS",
    "BUILDINGNUM": "ADDRESS",
    "AGE": "AGE",
    "DRIVERLICENSENUM": "DRIVER_ID",
    "CREDITCARDNUMBER": "CREDIT_CARD",
}


def _is_japanese(example: dict) -> bool:
    language = str(example.get("language", ""))
    region = str(example.get("region", ""))
    return language in {"ja", "ja-JP", "jp"} and region in {"", "JP", "ja-JP"}


def _merge_compatible(text: str, entities: list[dict]) -> list[dict]:
    """Merge adjacent source annotations into the detector's coarser taxonomy.

    OpenPII annotates GIVENNAME/SURNAME and address components separately,
    whereas the proposed detector emits PERSON and ADDRESS spans. We therefore
    merge only consecutive annotations mapped to the same target type when the
    gap contains delimiter characters rather than lexical content.
    """
    if not entities:
        return []

    entities = sorted(entities, key=lambda x: (x["start"], x["end"]))
    merged = [entities[0].copy()]
    for current in entities[1:]:
        previous = merged[-1]
        gap = text[previous["end"] : current["start"]]
        can_merge = (
            previous["type"] == current["type"]
            and current["type"] in {"PERSON", "ADDRESS"}
            and re.fullmatch(r"[\s,，、.・\-－ー−]*", gap) is not None
        )
        if can_merge:
            previous["end"] = current["end"]
            previous["text"] = text[previous["start"] : previous["end"]]
        else:
            merged.append(current.copy())
    return merged


def convert_example(example: dict, index: int) -> dict:
    text = example["source_text"]
    entities: list[dict] = []
    for mask in example.get("privacy_mask", []):
        label = str(mask.get("label", "")).upper()
        target = LABEL_MAP.get(label)
        if target is None:
            continue
        start = int(mask["start"])
        end = int(mask["end"])
        entities.append(
            {
                "type": target,
                "text": text[start:end],
                "start": start,
                "end": end,
                "source_label": label,
            }
        )

    entities = _merge_compatible(text, entities)
    return {
        "id": str(example.get("uid", index)),
        "text": text,
        "entities": entities,
        "metadata": {
            "dataset": DATASET_NAME,
            "language": example.get("language"),
            "region": example.get("region"),
            "source_index": index,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/ai4privacy_ja_500.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the optional benchmark dependency: pip install datasets") from exc

    ds = load_dataset(DATASET_NAME, split=args.split)
    japanese_indices = [i for i, example in enumerate(ds) if _is_japanese(example)]
    if len(japanese_indices) < args.n:
        raise SystemExit(f"Only {len(japanese_indices)} Japanese examples were found; requested {args.n}.")

    rng = random.Random(args.seed)
    sampled_indices = rng.sample(japanese_indices, args.n)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for idx in sampled_indices:
            record = convert_example(ds[idx], idx)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": DATASET_NAME,
        "split": args.split,
        "n": args.n,
        "seed": args.seed,
        "japanese_pool_size": len(japanese_indices),
        "output": str(output),
        "label_map": LABEL_MAP,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
