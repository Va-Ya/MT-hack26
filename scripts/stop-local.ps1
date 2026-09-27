$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$record = Join-Path $taskRoot 'artifacts\managed-processes.json'
if (-not (Test-Path -LiteralPath $record)) { throw 'No managed local processes were recorded.' }
foreach ($entry in (Get-Content -LiteralPath $record -Raw | ConvertFrom-Json)) {
    $candidate = Get-Process -Id $entry.id -ErrorAction SilentlyContinue
    if ($candidate -and $candidate.StartTime.ToFileTimeUtc() -eq $entry.started) {
        Stop-Process -Id $candidate.Id
    }
}
