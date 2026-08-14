param(
    [string]$OutputDir = "results\ginza_ultrasafe_500"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$StateFile = Join-Path (Join-Path $RepoRoot $OutputDir) "power_state.json"

if (-not (Test-Path $StateFile)) {
    Write-Host "No saved GiNZA power state found. Nothing to restore."
    exit 0
}

$state = Get-Content $StateFile -Raw -Encoding UTF8 | ConvertFrom-Json
$originalGuid = [string]$state.original_guid
$safeGuid = [string]$state.safe_guid

Write-Host "Restoring original power scheme $originalGuid"
& powercfg /setactive $originalGuid | Out-Null
try { & powercfg /delete $safeGuid | Out-Null } catch {}
Remove-Item -Force $StateFile -ErrorAction SilentlyContinue
Write-Host "Power scheme restored."
