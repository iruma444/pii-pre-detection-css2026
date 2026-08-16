# Benchmark datasets

Generated benchmark files are intentionally excluded from Git.

## Ai4Privacy OpenPII 1.5M

Create the current Japanese 500-document development/diagnostic set with:

```bash
python -m benchmarks.ai4privacy_adapter --n 500 --seed 42
```

This writes:

- `datasets/ai4privacy_ja_500.jsonl`
- `datasets/ai4privacy_ja_500.manifest.json`

The adapter maps the source taxonomy to the coarser taxonomy used by the detector and preserves character spans. Adjacent GIVENNAME/SURNAME spans and adjacent address components are merged only when the intervening text consists of delimiters.

## Important evaluation status

The seed-42 500-document set above has been inspected during development. Front-end regex rules, GiNZA label mapping, Unicode EMAIL candidate generation, and the local-LLM refinement prompt were changed after examining diagnostics/examples from this set. It must therefore be treated as a **development/diagnostic set**, not as an untouched final held-out test set.

Aggregate results from this development set are preserved under `paper_results/`.

If a final held-out benchmark is created later, it must be sampled without overlap with this development set and must not be used to add sample-specific rules, dictionary entries, prompt examples, or blacklist exceptions after its results are inspected.

The source dataset is synthetic and distributed separately under its own license. This repository stores only code and experiment manifests/aggregate results, not the generated dataset itself.
