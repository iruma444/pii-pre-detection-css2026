# pii-pre-detection-css2026

Research code for the CSS 2026 study on lightweight pre-detection of personally identifiable information (PII) using deterministic rules, dictionaries, Japanese NER, and selective local-LLM refinement.

> Status: experimental scaffold. Numerical benchmark results are intentionally not hard-coded in this repository; they are generated from a fixed benchmark configuration.

## Research question

The repository is structured to test whether a layered detector can preserve PII-detection accuracy while reducing reliance on a local LLM. The proposed pipeline is:

```text
input
  -> regex / dictionary / GiNZA candidate extraction
  -> deterministic candidate cleanup
  -> selective local-LLM refinement for ambiguous candidates only
  -> overlap resolution
  -> masking
```

The benchmark compares the following ablations on the same held-out data:

1. `regex`
2. `regex_dict`
3. `regex_dict_ginza`
4. `proposed` = Regex + Dictionary + GiNZA + selective local LLM
5. `llm_only` = full-text local-LLM extraction using the same Ollama model

## Metrics

The benchmark reports:

- TP / FP / FN
- Precision / Recall / F1
- exact-span metrics
- relaxed-span metrics
- 95% bootstrap confidence intervals for overall F1
- per-PII-type metrics
- mean latency
- p50 latency
- p95 latency
- total runtime
- local-LLM call count
- local-LLM call rate
- total LLM inference time
- LLM failures
- GiNZA availability
- Python / OS / package / GPU / Ollama model environment metadata

Exact-span scoring is the primary boundary-sensitive metric. Relaxed scoring is reported separately so that boundary correction is not hidden by a permissive matching rule.

## Repository layout

```text
.
├── benchmarks/
│   ├── ai4privacy_adapter.py   # fixed-seed OpenPII -> benchmark JSONL
│   ├── benchmark.py            # ablation, accuracy, latency, LLM-call benchmark
│   ├── environment.py          # reproducibility metadata
│   └── evaluator.py            # exact/relaxed span evaluation + bootstrap CI
├── configs/
│   └── benchmark.yaml
├── datasets/
│   └── README.md
├── src/
│   ├── pipeline.py
│   ├── schemas.py
│   └── extractors/
│       ├── regex_extractor.py
│       ├── dict_extractor.py
│       ├── nlp_extractor.py
│       ├── llm_extractor.py
│       └── llm_only_extractor.py
├── tests/
├── requirements.txt
└── README.md
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

For local-LLM experiments, install and start Ollama separately, then pull the model configured in `configs/benchmark.yaml`:

```bash
ollama pull qwen2.5:7b
```

The proposed and `llm_only` conditions use the same model and temperature setting.

## Build the held-out 500-example benchmark

```bash
python -m benchmarks.ai4privacy_adapter --n 500 --seed 42
```

The adapter uses the span annotations in Ai4Privacy OpenPII 1.5M and writes a local JSONL benchmark plus a manifest. Generated dataset files are excluded from Git.

The evaluation set must **not** be used to add sample-specific regex rules, dictionary terms, blacklist entries, prompt examples, or other exceptions. Error-driven development should use a separate development set.

## Run a smoke test

Run the non-LLM methods on a few examples first:

```bash
python -m benchmarks.benchmark --limit 10 --method regex --method regex_dict --method regex_dict_ginza
```

Then run the complete experiment:

```bash
python -m benchmarks.benchmark
```

Results are written to:

```text
results/
├── summary.csv
├── per_type.csv
├── per_document.jsonl
└── environment.json
```

## Interpretation of LLM call rate

`llm_only` sends every document to the local model, so its document-level call rate should be 1.0 unless calls fail before invocation. The proposed method invokes the model only when at least one ambiguous candidate survives the deterministic extraction stages. This makes LLM call rate and latency direct measurements of the claimed lightweight architecture rather than assumptions.

## Important methodological limitation

The local LLM in the proposed pipeline is a **refiner**, not an unrestricted second detector. It cannot recover a PII entity that all upstream candidate extractors fail to propose. Therefore, the recall ceiling of the proposed architecture remains dependent on the candidate-generation stage. The benchmark is designed to expose this limitation rather than hide it.

## Provenance

This research repository was derived from the experimental PII detector developed in `iruma444/AI-Librarium/pii-detector`, but sample-specific exception rules were intentionally excluded from this benchmark implementation to reduce evaluation leakage.
