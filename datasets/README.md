# Benchmark datasets

Generated benchmark files are intentionally excluded from Git.

## Ai4Privacy OpenPII 1.5M

Create the held-out Japanese benchmark with:

```bash
python -m benchmarks.ai4privacy_adapter --n 500 --seed 42
```

This writes:

- `datasets/ai4privacy_ja_500.jsonl`
- `datasets/ai4privacy_ja_500.manifest.json`

The adapter maps the source taxonomy to the coarser taxonomy used by the detector and preserves character spans. Adjacent GIVENNAME/SURNAME spans and adjacent address components are merged only when the intervening text consists of delimiters.

The final held-out benchmark must not be used to add sample-specific rules, dictionary entries, prompt examples, or blacklist exceptions. Development and error-driven tuning should be performed on a separate development dataset.

The source dataset is synthetic and distributed separately under its own license. This repository stores only code and experiment manifests, not the dataset itself.
