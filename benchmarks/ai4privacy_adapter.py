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


def reservoir_sample_japanese(stream, n: int, seed: int) -> tuple[list[tuple[int, dict]], int]:
    rng = random.Random(seed)
    reservoir: list[tuple[int, dict]] = []
    japanese_count = 0

    for source_index, example in enumerate(stream):
        if not _is_japanese(example):
            continue

        japanese_count += 1
        item = (source_index, dict(example))
        if len(reservoir) < n:
            reservoir.append(item)
            continue

        replacement_index = rng.randrange(japanese_count)
        if replacement_index < n:
            reservoir[replacement_index] = item

    reservoir.sort(key=lambda pair: pair[0])
    return reservoir, japanese_count


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
        raise SystemExit("Install the benchmark dependency: pip install datasets") from exc

    # Streaming avoids materializing the multi-GB dataset locally. The one-pass
    # reservoir sampler remains deterministic for a fixed dataset revision/order.
    stream = load_dataset(DATASET_NAME, split=args.split, streaming=True)
    sampled, japanese_pool_size = reservoir_sample_japanese(stream, args.n, args.seed)
    if japanese_pool_size < args.n:
        raise SystemExit(
            f"Only {japanese_pool_size} Japanese examples were found; requested {args.n}."
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for source_index, example in sampled:
            record = convert_example(example, source_index)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": DATASET_NAME,
        "split": args.split,
        "n": args.n,
        "seed": args.seed,
        "sampling": "one-pass reservoir sampling over streaming split",
        "japanese_pool_size": japanese_pool_size,
        "source_indices": [index for index, _ in sampled],
        "sample_ids": [str(example.get("uid", index)) for index, example in sampled],
        "output": str(output),
        "label_map": LABEL_MAP,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
