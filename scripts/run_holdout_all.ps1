param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if (-not (Test-Path $Python)) {
    Write-Host "[info] $Python not found; falling back to python on PATH."
    $Python = "python"
}

$Root = (Get-Location).Path
$ResultRoot = Join-Path $Root "results\holdout500"
New-Item -ItemType Directory -Force -Path $ResultRoot | Out-Null

$Transcript = Join-Path $ResultRoot "overnight_transcript.txt"
Start-Transcript -Path $Transcript -Append | Out-Null

try {
    Write-Host "============================================================"
    Write-Host " CSS2026 HOLDOUT-500 FORMAL RUN"
    Write-Host "============================================================"
    Write-Host "[python] $Python"
    Write-Host "[cwd]    $Root"

    # 0) Basic checks
    & $Python --version
    if ($LASTEXITCODE -ne 0) { throw "Python check failed." }

    & $Python -c "import yaml, datasets, spacy; import ja_ginza; print('python deps: OK')"
    if ($LASTEXITCODE -ne 0) { throw "Python dependencies / ja_ginza check failed." }

    try {
        $ollamaModels = & ollama list 2>&1
        if ($LASTEXITCODE -ne 0) { throw "ollama list failed" }
        $ollamaModels | Out-Host
        if (($ollamaModels -join "`n") -notmatch "gpt-oss:20b") {
            throw "gpt-oss:20b is not installed in Ollama."
        }
    }
    catch {
        throw "Ollama check failed. Make sure Ollama is running and gpt-oss:20b is installed. Details: $_"
    }

    # 1) Create a true holdout set, excluding the current development 500.
    if (-not (Test-Path "datasets\ai4privacy_ja_holdout_500.jsonl")) {
        Write-Host ""
        Write-Host "=== [1/8] CREATE HOLDOUT 500 ==="
        & $Python -m benchmarks.make_holdout `
            --dev "datasets/ai4privacy_ja_500.jsonl" `
            --dev-manifest "datasets/ai4privacy_ja_500.manifest.json" `
            --output "datasets/ai4privacy_ja_holdout_500.jsonl" `
            --n 500 `
            --seed 20260817 `
            --holdout-config "configs/benchmark_holdout.yaml"
        if ($LASTEXITCODE -ne 0) { throw "Holdout generation failed." }
    }
    else {
        Write-Host ""
        Write-Host "=== [1/8] HOLDOUT ALREADY EXISTS: reusing frozen file ==="
        if (-not (Test-Path "configs\benchmark_holdout.yaml")) {
            throw "Holdout JSONL exists but configs/benchmark_holdout.yaml is missing. Do not regenerate blindly."
        }
    }

    # 2) R
    Write-Host ""
    Write-Host "=== [2/8] R ==="
    & $Python -m benchmarks.benchmark `
        --config "configs/benchmark_holdout.yaml" `
        --output-dir "results/holdout500/R" `
        --method regex
    if ($LASTEXITCODE -ne 0) { throw "R benchmark failed." }

    # 3) R+D
    Write-Host ""
    Write-Host "=== [3/8] R+D ==="
    & $Python -m benchmarks.benchmark `
        --config "configs/benchmark_holdout.yaml" `
        --output-dir "results/holdout500/R_D" `
        --method regex_dict
    if ($LASTEXITCODE -ne 0) { throw "R+D benchmark failed." }

    # 4) R+D+G
    Write-Host ""
    Write-Host "=== [4/8] R+D+G ==="
    & $Python -m benchmarks.benchmark `
        --config "configs/benchmark_holdout.yaml" `
        --output-dir "results/holdout500/R_D_G" `
        --method regex_dict_ginza
    if ($LASTEXITCODE -ne 0) { throw "R+D+G benchmark failed." }

    # 5) Proposed - durable/resumable
    Write-Host ""
    Write-Host "=== [5/8] PROPOSED (durable / resumable) ==="
    & $Python -m benchmarks.durable_proposed `
        --config "configs/benchmark_holdout.yaml" `
        --output "results/holdout500/Proposed/live/per_document.jsonl" `
        --think-level low `
        --num-ctx 4096 `
        --timeout-seconds 180
    if ($LASTEXITCODE -ne 0) { throw "Proposed durable benchmark failed." }

    & $Python -m benchmarks.aggregate_durable_results `
        --config "configs/benchmark_holdout.yaml" `
        --input "results/holdout500/Proposed/live/per_document.jsonl" `
        --output-dir "results/holdout500/Proposed/final" `
        --expected-documents 500
    if ($LASTEXITCODE -ne 0) { throw "Proposed aggregation failed." }

    # 6) LLM-only - durable/resumable
    Write-Host ""
    Write-Host "=== [6/8] LLM-ONLY (durable / resumable) ==="
    & $Python -m benchmarks.durable_llm_only `
        --config "configs/benchmark_holdout.yaml" `
        --output "results/holdout500/LLM_only/live/per_document.jsonl" `
        --think-level low `
        --num-ctx 4096 `
        --timeout-seconds 180
    if ($LASTEXITCODE -ne 0) { throw "LLM-only durable benchmark failed." }

    & $Python -m benchmarks.aggregate_durable_results `
        --config "configs/benchmark_holdout.yaml" `
        --input "results/holdout500/LLM_only/live/per_document.jsonl" `
        --output-dir "results/holdout500/LLM_only/final" `
        --expected-documents 500
    if ($LASTEXITCODE -ne 0) { throw "LLM-only aggregation failed." }

    # 7) Paired bootstrap: R+D+G vs Proposed
    Write-Host ""
    Write-Host "=== [7/8] PAIRED BOOTSTRAP: R+D+G vs Proposed ==="
    & $Python -m benchmarks.paired_bootstrap `
        --config "configs/benchmark_holdout.yaml" `
        --baseline "results/holdout500/R_D_G/per_document.jsonl" `
        --candidate "results/holdout500/Proposed/live/per_document.jsonl" `
        --iterations 2000 `
        --seed 42 `
        --output "results/holdout500/paired_R_D_G_vs_Proposed.csv"
    if ($LASTEXITCODE -ne 0) { throw "Paired bootstrap failed." }

    # 8) Print the files we need for the paper.
    Write-Host ""
    Write-Host "=== [8/8] FINAL RESULTS ==="

    $SummaryFiles = @(
        "results/holdout500/R/summary.csv",
        "results/holdout500/R_D/summary.csv",
        "results/holdout500/R_D_G/summary.csv",
        "results/holdout500/Proposed/final/summary.csv",
        "results/holdout500/LLM_only/final/summary.csv"
    )

    foreach ($file in $SummaryFiles) {
        Write-Host ""
        Write-Host "----- $file -----"
        Get-Content $file | Out-Host
    }

    Write-Host ""
    Write-Host "----- Proposed per type -----"
    Get-Content "results/holdout500/Proposed/final/per_type.csv" | Out-Host

    Write-Host ""
    Write-Host "----- LLM-only per type -----"
    Get-Content "results/holdout500/LLM_only/final/per_type.csv" | Out-Host

    Write-Host ""
    Write-Host "----- Paired bootstrap -----"
    Get-Content "results/holdout500/paired_R_D_G_vs_Proposed.csv" | Out-Host

    Write-Host ""
    Write-Host "============================================================"
    Write-Host " ALL DONE"
    Write-Host " Results: $ResultRoot"
    Write-Host " Transcript: $Transcript"
    Write-Host "============================================================"
}
finally {
    Stop-Transcript | Out-Null
}
