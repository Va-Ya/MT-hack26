$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    throw 'Create .venv and install requirements-lock.txt first (see README).'
}
foreach ($port in @(8000,5173)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use. Existing dashboard: http://127.0.0.1:5173"
    }
}
New-Item -ItemType Directory -Path artifacts -Force | Out-Null
$backend = Start-Process -FilePath (Join-Path $taskRoot '.venv\Scripts\python.exe') -ArgumentList '-m','uvicorn','backend.app:app','--host','127.0.0.1','--port','8000' -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRoot 'artifacts\backend.log') -RedirectStandardError (Join-Path $taskRoot 'artifacts\backend-error.log') -PassThru
$frontend = Start-Process -FilePath (Get-Command node).Source -ArgumentList 'node_modules/vite/bin/vite.js','--host','127.0.0.1','--port','5173' -WorkingDirectory (Join-Path $taskRoot 'dashboard') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRoot 'artifacts\dashboard.log') -RedirectStandardError (Join-Path $taskRoot 'artifacts\dashboard-error.log') -PassThru
@(@{id=$backend.Id;started=$backend.StartTime.ToFileTimeUtc()},@{id=$frontend.Id;started=$frontend.StartTime.ToFileTimeUtc()}) | ConvertTo-Json | Set-Content -Encoding utf8 artifacts\managed-processes.json
Write-Output 'Dashboard: http://127.0.0.1:5173 | Swagger: http://127.0.0.1:8000/docs'
Write-Output 'Now start replay using the command in README.'
