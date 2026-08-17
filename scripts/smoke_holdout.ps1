param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if (-not (Test-Path $Python)) {
    Write-Host "[info] $Python not found; falling back to python on PATH."
    $Python = "python"
}

$SmokeRoot = "results\smoke_holdout"
New-Item -ItemType Directory -Force -Path $SmokeRoot | Out-Null

Write-Host "============================================================"
Write-Host " CSS2026 HOLDOUT SMOKE TEST"
Write-Host "============================================================"

# 0) Syntax / dependency / model checks
& $Python -m py_compile benchmarks\make_holdout.py benchmarks\durable_proposed.py benchmarks\durable_llm_only.py benchmarks\paired_bootstrap.py
if ($LASTEXITCODE -ne 0) { throw "Python syntax check failed." }

& $Python -c "import yaml, datasets, spacy; import ja_ginza; print('python deps: OK')"
if ($LASTEXITCODE -ne 0) { throw "Python dependencies / ja_ginza check failed." }

$ollamaModels = & ollama list 2>&1
if ($LASTEXITCODE -ne 0) { throw "ollama list failed" }
$ollamaModels | Out-Host
if (($ollamaModels -join "`n") -notmatch "gpt-oss:20b") {
    throw "gpt-oss:20b is not installed in Ollama."
}

# 1) Test holdout generation on a TEMPORARY 5-document sample.
# This is NOT the formal holdout set and uses a different seed/path.
Remove-Item "datasets\_smoke_holdout_5.jsonl" -ErrorAction SilentlyContinue
Remove-Item "datasets\_smoke_holdout_5.manifest.json" -ErrorAction SilentlyContinue
Remove-Item "configs\_smoke_holdout.yaml" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== [1/7] temporary holdout generation (5 docs) ==="
& $Python -m benchmarks.make_holdout `
    --dev "datasets/ai4privacy_ja_500.jsonl" `
    --dev-manifest "datasets/ai4privacy_ja_500.manifest.json" `
    --output "datasets/_smoke_holdout_5.jsonl" `
    --n 5 `
    --seed 20260816 `
    --holdout-config "configs/_smoke_holdout.yaml"
if ($LASTEXITCODE -ne 0) { throw "Temporary holdout generation failed." }

# 2-4) Deterministic methods on only 5 docs
Write-Host ""
Write-Host "=== [2/7] R ==="
& $Python -m benchmarks.benchmark --config "configs/_smoke_holdout.yaml" --output-dir "$SmokeRoot\R" --method regex
if ($LASTEXITCODE -ne 0) { throw "R smoke test failed." }

Write-Host ""
Write-Host "=== [3/7] R+D ==="
& $Python -m benchmarks.benchmark --config "configs/_smoke_holdout.yaml" --output-dir "$SmokeRoot\R_D" --method regex_dict
if ($LASTEXITCODE -ne 0) { throw "R+D smoke test failed." }

Write-Host ""
Write-Host "=== [4/7] R+D+G ==="
& $Python -m benchmarks.benchmark --config "configs/_smoke_holdout.yaml" --output-dir "$SmokeRoot\R_D_G" --method regex_dict_ginza
if ($LASTEXITCODE -ne 0) { throw "R+D+G smoke test failed." }

# 5) Proposed on 2 docs only; fresh output every smoke run
Write-Host ""
Write-Host "=== [5/7] Proposed (2 docs) ==="
Remove-Item "$SmokeRoot\proposed.jsonl" -ErrorAction SilentlyContinue
& $Python -m benchmarks.durable_proposed `
    --config "configs/_smoke_holdout.yaml" `
    --output "$SmokeRoot/proposed.jsonl" `
    --limit 2 `
    --think-level low `
    --num-ctx 4096 `
    --timeout-seconds 180
if ($LASTEXITCODE -ne 0) { throw "Proposed smoke test failed." }

& $Python -m benchmarks.aggregate_durable_results `
    --config "configs/_smoke_holdout.yaml" `
    --input "$SmokeRoot/proposed.jsonl" `
    --output-dir "$SmokeRoot\Proposed_final" `
    --expected-documents 2
if ($LASTEXITCODE -ne 0) { throw "Proposed smoke aggregation failed." }

# 6) LLM-only on 1 doc only
Write-Host ""
Write-Host "=== [6/7] LLM-only (1 doc) ==="
Remove-Item "$SmokeRoot\llm_only.jsonl" -ErrorAction SilentlyContinue
& $Python -m benchmarks.durable_llm_only `
    --config "configs/_smoke_holdout.yaml" `
    --output "$SmokeRoot/llm_only.jsonl" `
    --limit 1 `
    --think-level low `
    --num-ctx 4096 `
    --timeout-seconds 180
if ($LASTEXITCODE -ne 0) { throw "LLM-only smoke test failed." }

& $Python -m benchmarks.aggregate_durable_results `
    --config "configs/_smoke_holdout.yaml" `
    --input "$SmokeRoot/llm_only.jsonl" `
    --output-dir "$SmokeRoot\LLM_only_final" `
    --expected-documents 1
if ($LASTEXITCODE -ne 0) { throw "LLM-only smoke aggregation failed." }

# 7) Basic structural checks
Write-Host ""
Write-Host "=== [7/7] structural checks ==="
& $Python -c "import json, pathlib; p=pathlib.Path('datasets/_smoke_holdout_5.manifest.json'); m=json.loads(p.read_text(encoding='utf-8')); assert m['overlap_with_development_ids']==0; assert m['overlap_with_development_source_indices']==0; assert m['n']==5; print('manifest overlap check: PASS')"
if ($LASTEXITCODE -ne 0) { throw "Manifest overlap check failed." }

Write-Host ""
Write-Host "============================================================"
Write-Host " SMOKE TEST PASS"
Write-Host " Do NOT use the temporary 5 documents for the paper."
Write-Host " Next: .\scripts\run_holdout_all.ps1"
Write-Host "============================================================"
