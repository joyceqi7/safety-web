$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (!(Test-Path '.venv\Scripts\python.exe')) {
    py -3.13 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required.' }
    & '.\.venv\Scripts\python.exe' -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
    if ($LASTEXITCODE -ne 0) { throw 'Torch installation failed.' }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.lock.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
}
if (!(Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    (Get-Content '.env' -Raw).Replace('APP_ENV=production','APP_ENV=development') | Set-Content -Encoding utf8 '.env'
}
Write-Host 'Open http://127.0.0.1:8765'
& '.\.venv\Scripts\python.exe' -m uvicorn app.main:app --host 127.0.0.1 --port 8765
