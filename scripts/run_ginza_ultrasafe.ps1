param(
    [int]$MaxCpuPercent = 35,
    [double]$PauseSeconds = 5.0,
    [string]$AffinityMask = "1",
    [string]$OutputDir = "results\ginza_ultrasafe_500"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$OutputRoot = Join-Path $RepoRoot $OutputDir
$ChunksDir = Join-Path $OutputRoot "chunks"
$Predictions = Join-Path $ChunksDir "live\per_document.jsonl"
$StateFile = Join-Path $OutputRoot "power_state.json"
$BaseConfig = Join-Path $RepoRoot "configs\benchmark.yaml"

if (-not (Test-Path $Python)) {
    throw "Python venv not found: $Python"
}
if ($MaxCpuPercent -lt 10 -or $MaxCpuPercent -gt 100) {
    throw "MaxCpuPercent must be between 10 and 100"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

# Native numerical libraries stay single-threaded inside the child process.
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"

function Get-GuidFromText([string]$Text) {
    $match = [regex]::Match($Text, '[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}')
    if (-not $match.Success) {
        throw "Could not parse power scheme GUID from: $Text"
    }
    return $match.Value
}

function Invoke-LowAffinityPython([string[]]$Arguments) {
    $quoted = @('"' + $Python + '"') + ($Arguments | ForEach-Object {
        if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    })
    $cmdLine = 'start "" /wait /b /low /affinity {0} {1}' -f $AffinityMask, ($quoted -join ' ')
    Push-Location $RepoRoot
    try {
        & cmd.exe /c $cmdLine
        if ($LASTEXITCODE -ne 0) {
            throw "Child Python exited with code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

$originalGuid = $null
$safeGuid = $null

if (Test-Path $StateFile) {
    $state = Get-Content $StateFile -Raw -Encoding UTF8 | ConvertFrom-Json
    $originalGuid = [string]$state.original_guid
    $safeGuid = [string]$state.safe_guid
    Write-Host "[power] reusing crash-safe scheme $safeGuid"
}
else {
    $activeText = (& powercfg /getactivescheme | Out-String)
    $originalGuid = Get-GuidFromText $activeText
    $duplicateText = (& powercfg /duplicatescheme $originalGuid | Out-String)
    $safeGuid = Get-GuidFromText $duplicateText

    @{
        original_guid = $originalGuid
        safe_guid = $safeGuid
        created_at = (Get-Date).ToString('o')
    } | ConvertTo-Json | Set-Content $StateFile -Encoding UTF8

    Write-Host "[power] original scheme: $originalGuid"
    Write-Host "[power] created safe scheme: $safeGuid"
}

# Modify only the duplicated power scheme.  The user's original scheme remains untouched.
& powercfg /setactive $safeGuid | Out-Null
& powercfg /setacvalueindex $safeGuid SUB_PROCESSOR PROCTHROTTLEMIN 5 | Out-Null
& powercfg /setacvalueindex $safeGuid SUB_PROCESSOR PROCTHROTTLEMAX $MaxCpuPercent | Out-Null
& powercfg /setacvalueindex $safeGuid SUB_PROCESSOR PERFBOOSTMODE 0 | Out-Null
& powercfg /setactive $safeGuid | Out-Null

Write-Host "[power] max CPU performance=${MaxCpuPercent}% boost=disabled"
Write-Host "[power] child affinity=$AffinityMask priority=low pause=${PauseSeconds}s"
Write-Host "[power] if the PC reboots, this capped scheme remains active; rerun this script to resume."

$completed = $false
try {
    Invoke-LowAffinityPython @(
        '-m', 'benchmarks.durable_ginza',
        '--config', $BaseConfig,
        '--output', $Predictions,
        '--pause-seconds', [string]$PauseSeconds
    )

    Write-Host "[aggregate] 500-document official metrics under the same power cap"
    Invoke-LowAffinityPython @(
        '-m', 'benchmarks.aggregate_chunk_results',
        '--chunks-dir', $ChunksDir,
        '--output-dir', $OutputRoot,
        '--config', $BaseConfig,
        '--expected-documents', '500'
    )
    $completed = $true
}
finally {
    if ($completed) {
        Write-Host "[power] restoring original power scheme $originalGuid"
        & powercfg /setactive $originalGuid | Out-Null
        try { & powercfg /delete $safeGuid | Out-Null } catch {}
        Remove-Item -Force $StateFile -ErrorAction SilentlyContinue
        Write-Host "[done] $OutputRoot\summary.csv"
    }
    else {
        Write-Host "[resume] run did not finish; safe capped power scheme is intentionally left active."
        Write-Host "[resume] rerun .\scripts\run_ginza_ultrasafe.ps1 to continue from the last fsynced document."
        Write-Host "[restore] to stop and restore immediately, run .\scripts\restore_ginza_power.ps1"
    }
}
