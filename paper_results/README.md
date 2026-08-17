# CSS 2026 paper result snapshot

This directory preserves aggregate results used while revising the CSS 2026 manuscript.

## Status

The current 500-document Ai4Privacy Japanese benchmark is a **development/diagnostic set**, not a clean final held-out test set. During development, examples and aggregate diagnostics from this set were inspected and the front-end rules, GiNZA label mapping, Unicode EMAIL candidate generation, and local-LLM refinement prompt were changed in response. Therefore, do not describe these 500 documents as an untouched held-out test set in the paper.

The results are nevertheless retained as the fixed development result snapshot from which the manuscript is being revised.

## Dataset and evaluation

- Dataset config: `datasets/ai4privacy_ja_500.jsonl`
- Configured sample size: 500
- Configured seed: 42
- Target types: PERSON, EMAIL, PHONE, ADDRESS, CREDIT_CARD
- Exact and relaxed span evaluation are reported separately.
- Paired bootstrap: 2,000 document-level resamples, seed 42.

The generated JSONL dataset itself remains excluded from Git. See `datasets/README.md`.

## Proposed-method inference condition

- Local model: `gpt-oss:20b`
- Backend: Ollama
- temperature: 0.0
- think level: low
- context length (`num_ctx`): 4096
- per-request timeout: 180 s
- documents: 500
- LLM calls: 426 / 500 documents (85.2%)
- LLM failures: 0

## Main development-set results

| Method | Exact F1 | Relaxed F1 | Mean latency |
|---|---:|---:|---:|
| Regex | 0.3619 | 0.4912 | 0.08 ms/doc |
| Regex+Dictionary | 0.3414 | 0.5665 | 0.30 ms/doc |
| Regex+Dictionary+GiNZA | 0.3411 | 0.6521 | 24.84 ms/doc |
| Proposed | **0.4488** | **0.7919** | **12.28 s/doc** |
| LLM-only reference | 0.3887 | 0.6058 | 44.73 s/doc |

For Proposed, Exact F1 95% bootstrap CI is [0.4252, 0.4722], and Relaxed F1 95% bootstrap CI is [0.7773, 0.8073].

Compared with Regex+Dictionary+GiNZA, Proposed improves Exact F1 by 0.1077 (paired-bootstrap 95% CI [0.0947, 0.1207]) and Relaxed F1 by 0.1398 ([0.1264, 0.1536]).

The strongest gains are on EMAIL (Exact +0.4554; Relaxed +0.5267) and PERSON (Exact +0.0545; Relaxed +0.1725). PHONE and CREDIT_CARD are unchanged, while ADDRESS shows only a small Exact improvement and no clear Relaxed improvement.

## Files

- `dev500_overall.csv`: aggregate method results and runtime/reference information.
- `dev500_per_type_proposed.csv`: Proposed per-type precision/recall/F1.
- `dev500_paired_vs_ginza.csv`: paired bootstrap comparison between Regex+Dictionary+GiNZA and Proposed.

## Code snapshot

The development result snapshot was finalized after the candidate-constrained Unicode EMAIL path and LLM prompt were fixed. The `research-benchmark` branch HEAD immediately before saving these result files was:

`fc5196ba4c0110a85963080c24aa8f7141a835ef`

Paper edits should preserve the distinction between this development-set evidence and any future untouched final evaluation.
