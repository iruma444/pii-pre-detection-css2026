from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import yaml

from benchmarks.ai4privacy_adapter import DATASET_NAME, _is_japanese, convert_example


def load_jsonl_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ids.add(str(row["id"]))
    return ids


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a 500-document holdout set that excludes the existing development set."
    )
    parser.add_argument("--dev", default="datasets/ai4privacy_ja_500.jsonl")
    parser.add_argument(
        "--dev-manifest",
        default="datasets/ai4privacy_ja_500.manifest.json",
    )
    parser.add_argument(
        "--output",
        default="datasets/ai4privacy_ja_holdout_500.jsonl",
    )
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--revision",
        default=None,
        help="Immutable Hugging Face dataset revision. If omitted, use the dev manifest's resolved_revision.",
    )
    parser.add_argument(
        "--base-config",
        default="configs/benchmark.yaml",
    )
    parser.add_argument(
        "--holdout-config",
        default="configs/benchmark_holdout.yaml",
    )
    args = parser.parse_args()

    dev_path = Path(args.dev)
    dev_manifest_path = Path(args.dev_manifest)
    output_path = Path(args.output)

    if not dev_path.exists():
        raise SystemExit(f"Development dataset not found: {dev_path}")

    dev_ids = load_jsonl_ids(dev_path)

    manifest: dict = {}
    if dev_manifest_path.exists():
        manifest = json.loads(dev_manifest_path.read_text(encoding="utf-8"))

    revision = args.revision or manifest.get("resolved_revision")
    if not revision:
        raise SystemExit(
            "Could not determine the immutable dataset revision. "
            "Expected resolved_revision in the development manifest. "
            "Pass --revision <commit-sha> explicitly."
        )

    dev_source_indices = {int(x) for x in manifest.get("source_indices", [])}

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing benchmark dependency. Activate the benchmark venv and run "
            "`pip install -r requirements.txt`."
        ) from exc

    print(f"[holdout] dataset={DATASET_NAME}")
    print(f"[holdout] revision={revision}")
    print(f"[holdout] dev_ids={len(dev_ids)}")
    print(f"[holdout] dev_source_indices={len(dev_source_indices)}")
    print(f"[holdout] target_n={args.n} seed={args.seed}")

    stream = load_dataset(
        DATASET_NAME,
        split=args.split,
        streaming=True,
        revision=revision,
    )

    rng = random.Random(args.seed)
    reservoir: list[tuple[int, dict]] = []
    eligible_count = 0
    japanese_count = 0
    excluded_dev_count = 0

    for source_index, example in enumerate(stream):
        if not _is_japanese(example):
            continue

        japanese_count += 1
        example_id = str(example.get("uid", source_index))

        if example_id in dev_ids or source_index in dev_source_indices:
            excluded_dev_count += 1
            continue

        eligible_count += 1
        item = (source_index, dict(example))

        if len(reservoir) < args.n:
            reservoir.append(item)
            continue

        replacement_index = rng.randrange(eligible_count)
        if replacement_index < args.n:
            reservoir[replacement_index] = item

    if eligible_count < args.n:
        raise SystemExit(
            f"Only {eligible_count} eligible Japanese documents remained after exclusion; "
            f"requested {args.n}."
        )

    reservoir.sort(key=lambda pair: pair[0])
    holdout_source_indices = [index for index, _ in reservoir]
    holdout_ids = [
        str(example.get("uid", source_index))
        for source_index, example in reservoir
    ]

    overlap_ids = dev_ids.intersection(holdout_ids)
    overlap_indices = dev_source_indices.intersection(holdout_source_indices)
    if overlap_ids or overlap_indices:
        raise SystemExit(
            "HOLDOUT OVERLAP DETECTED: "
            f"id_overlap={len(overlap_ids)}, source_index_overlap={len(overlap_indices)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for source_index, example in reservoir:
            record = convert_example(example, source_index)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    output_manifest = {
        "dataset": DATASET_NAME,
        "resolved_revision": revision,
        "split": args.split,
        "role": "held-out test",
        "n": args.n,
        "seed": args.seed,
        "sampling": "reservoir sampling from Japanese documents after excluding development ids/source indices",
        "japanese_pool_size": japanese_count,
        "eligible_after_dev_exclusion": eligible_count,
        "excluded_development_documents_seen": excluded_dev_count,
        "development_dataset": str(dev_path),
        "development_manifest": str(dev_manifest_path),
        "development_document_count": len(dev_ids),
        "source_indices": holdout_source_indices,
        "sample_ids": holdout_ids,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "overlap_with_development_ids": 0,
        "overlap_with_development_source_indices": 0,
    }

    manifest_out = output_path.with_suffix(".manifest.json")
    manifest_out.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Create a separate benchmark config so the development config is never overwritten.
    base_cfg_path = Path(args.base_config)
    holdout_cfg_path = Path(args.holdout_config)
    cfg = yaml.safe_load(base_cfg_path.read_text(encoding="utf-8"))
    cfg["dataset"]["path"] = str(output_path).replace("\\", "/")
    cfg["dataset"]["sample_size"] = args.n
    cfg["dataset"]["seed"] = args.seed
    cfg["dataset"]["role"] = "held-out test"
    cfg["dataset"]["resolved_revision"] = revision
    holdout_cfg_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    print("[holdout] PASS: development/holdout overlap = 0")
    print(f"[holdout] output={output_path}")
    print(f"[holdout] manifest={manifest_out}")
    print(f"[holdout] sha256={output_manifest['output_sha256']}")
    print(f"[holdout] config={holdout_cfg_path}")
    print(
        "[holdout] IMPORTANT: Freeze regex/dictionary/GiNZA mapping/LLM prompt/settings now. "
        "Do not tune the method using this holdout set."
    )


if __name__ == "__main__":
    main()
