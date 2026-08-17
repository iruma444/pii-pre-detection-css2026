param(
    [int]$ChunkSize = 5,
    [int]$PauseSeconds = 10,
    [string]$AffinityMask = "1",
    [string]$OutputDir = "results\ginza_safe_500"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Dataset = Join-Path $RepoRoot "datasets\ai4privacy_ja_500.jsonl"
$BaseConfig = Join-Path $RepoRoot "configs\benchmark.yaml"
$RootOutput = Join-Path $RepoRoot $OutputDir
$ChunksDir = Join-Path $RootOutput "chunks"
$TempDir = Join-Path $RootOutput "_tmp"

if (-not (Test-Path $Python)) {
    throw "Python venv not found: $Python"
}
if (-not (Test-Path $Dataset)) {
    throw "Dataset not found: $Dataset"
}

New-Item -ItemType Directory -Force -Path $ChunksDir, $TempDir | Out-Null

# Keep native numeric libraries single-threaded. These settings apply to each child process.
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$AllLines = [System.IO.File]::ReadAllLines($Dataset, $Utf8NoBom)
$Total = $AllLines.Count

Write-Host "[safe-ginza] total=$Total chunk_size=$ChunkSize pause=${PauseSeconds}s affinity=$AffinityMask"
Write-Host "[safe-ginza] completed chunks are skipped automatically after a reboot/restart."

for ($Start = 0; $Start -lt $Total; $Start += $ChunkSize) {
    $EndExclusive = [Math]::Min($Start + $ChunkSize, $Total)
    $Expected = $EndExclusive - $Start
    $ChunkName = "{0:D4}_{1:D4}" -f $Start, ($EndExclusive - 1)
    $ChunkDir = Join-Path $ChunksDir $ChunkName
    $PerDocument = Join-Path $ChunkDir "per_document.jsonl"
    $Summary = Join-Path $ChunkDir "summary.csv"

    $Complete = $false
    if ((Test-Path $Summary) -and (Test-Path $PerDocument)) {
        try {
            $LineCount = ([System.IO.File]::ReadAllLines($PerDocument, $Utf8NoBom)).Count
            if ($LineCount -eq $Expected) {
                $Complete = $true
            }
        }
        catch {
            $Complete = $false
        }
    }

    if ($Complete) {
        Write-Host "[skip] $ChunkName already complete"
        continue
    }

    if (Test-Path $ChunkDir) {
        Remove-Item -Recurse -Force $ChunkDir
    }
    New-Item -ItemType Directory -Force -Path $ChunkDir | Out-Null

    $ChunkDataset = Join-Path $TempDir ("dataset_{0}.jsonl" -f $ChunkName)
    $ChunkConfig = Join-Path $TempDir ("config_{0}.yaml" -f $ChunkName)

    $Slice = New-Object System.Collections.Generic.List[string]
    for ($i = $Start; $i -lt $EndExclusive; $i++) {
        $Slice.Add($AllLines[$i])
    }
    [System.IO.File]::WriteAllLines($ChunkDataset, [string[]]$Slice, $Utf8NoBom)

    $ConfigText = [System.IO.File]::ReadAllText($BaseConfig, $Utf8NoBom)
    $DatasetYamlPath = $ChunkDataset.Replace("\", "/")
    $ConfigText = [regex]::Replace(
        $ConfigText,
        '(?m)^  path:\s*.*$',
        ('  path: "' + $DatasetYamlPath + '"'),
        1
    )
    # Chunk-level confidence intervals are discarded. Keep this tiny to minimize CPU load;
    # the final aggregator recomputes the official 2000-iteration CI on all 500 documents.
    $ConfigText = [regex]::Replace(
        $ConfigText,
        '(?m)^  iterations:\s*\d+\s*$',
        '  iterations: 10',
        1
    )
    [System.IO.File]::WriteAllText($ChunkConfig, $ConfigText, $Utf8NoBom)

    Write-Host "[run] $ChunkName ($Expected docs)"

    $CmdLine = 'start "" /wait /b /low /affinity {0} "{1}" -m benchmarks.benchmark --config "{2}" --method regex_dict_ginza --output-dir "{3}"' -f `
        $AffinityMask, $Python, $ChunkConfig, $ChunkDir

    Push-Location $RepoRoot
    try {
        & cmd.exe /c $CmdLine
    }
    finally {
        Pop-Location
    }

    if (-not ((Test-Path $Summary) -and (Test-Path $PerDocument))) {
        throw "Chunk $ChunkName did not complete. Re-run this script after recovery; completed chunks will be skipped."
    }

    $Produced = ([System.IO.File]::ReadAllLines($PerDocument, $Utf8NoBom)).Count
    if ($Produced -ne $Expected) {
        throw "Chunk $ChunkName produced $Produced rows; expected $Expected. Re-run after recovery."
    }

    Write-Host "[cooldown] ${PauseSeconds}s"
    Start-Sleep -Seconds $PauseSeconds
}

Write-Host "[aggregate] recomputing official full-dataset metrics"
Push-Location $RepoRoot
try {
    & $Python -m benchmarks.aggregate_chunk_results `
        --chunks-dir $ChunksDir `
        --output-dir $RootOutput `
        --config $BaseConfig `
        --expected-documents $Total
}
finally {
    Pop-Location
}

Write-Host "[done] $RootOutput\summary.csv"
