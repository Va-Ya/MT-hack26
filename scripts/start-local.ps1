param([string]$NodeExe = '', [string]$PythonExe = '', [int]$BackendPort = 8000, [int]$FrontendPort = 5173)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
if (Test-Path -LiteralPath '.env') {
    foreach ($line in Get-Content -LiteralPath '.env' -Encoding UTF8) {
        if ($line -match '^\s*([A-Z][A-Z0-9_]*)\s*=\s*(.*?)\s*$') {
            $settingName = $Matches[1]
            $settingValue = $Matches[2].Trim('"').Trim("'")
            [Environment]::SetEnvironmentVariable($settingName, $settingValue, 'Process')
        }
    }
}
if (-not $NodeExe) {
    $nodeCommand = Get-Command node -ErrorAction SilentlyContinue
    if ($nodeCommand) { $NodeExe = $nodeCommand.Source }
    elseif (Test-Path -LiteralPath '.local-runtime.json') {
        $NodeExe = (Get-Content -LiteralPath '.local-runtime.json' -Raw | ConvertFrom-Json).node
    }
}
if (-not $NodeExe -or -not (Test-Path -LiteralPath $NodeExe)) { throw 'Node.js not found. Install Node 24 or pass -NodeExe with its absolute path.' }
if (-not $PythonExe) {
    if (Test-Path -LiteralPath '.venv\Scripts\python.exe') { $PythonExe = Join-Path $taskRoot '.venv\Scripts\python.exe' }
    elseif (Test-Path -LiteralPath '.local-runtime.json') { $PythonExe = (Get-Content -LiteralPath '.local-runtime.json' -Raw | ConvertFrom-Json).python }
}
if (-not $PythonExe -or -not (Test-Path -LiteralPath $PythonExe)) { throw 'Create .venv and install requirements-lock.txt, or pass -PythonExe.' }
$env:BACKEND_PORT = [string]$BackendPort
$env:VITE_BACKEND_DOCS_URL = "http://127.0.0.1:$BackendPort/docs"
foreach ($port in @($BackendPort,$FrontendPort)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $port is already in use. Existing dashboard: http://127.0.0.1:5173"
    }
}
New-Item -ItemType Directory -Path artifacts -Force | Out-Null
$backend = Start-Process -FilePath $PythonExe -ArgumentList '-m','uvicorn','backend.app:app','--host','127.0.0.1','--port',([string]$BackendPort) -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRoot 'artifacts\backend.log') -RedirectStandardError (Join-Path $taskRoot 'artifacts\backend-error.log') -PassThru
$frontend = Start-Process -FilePath $NodeExe -ArgumentList 'node_modules/vite/bin/vite.js','--host','127.0.0.1','--port',([string]$FrontendPort),'--strictPort' -WorkingDirectory (Join-Path $taskRoot 'dashboard') -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRoot 'artifacts\dashboard.log') -RedirectStandardError (Join-Path $taskRoot 'artifacts\dashboard-error.log') -PassThru
@(@{id=$backend.Id;started=$backend.StartTime.ToFileTimeUtc()},@{id=$frontend.Id;started=$frontend.StartTime.ToFileTimeUtc()}) | ConvertTo-Json | Set-Content -Encoding utf8 artifacts\managed-processes.json
Write-Output "Dashboard: http://127.0.0.1:$FrontendPort | Swagger: http://127.0.0.1:$BackendPort/docs"
Write-Output 'Weather and what-if work without the dataset. Replay requires the original CSV files.'
