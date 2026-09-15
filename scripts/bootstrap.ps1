$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv venv --python 3.11 .venv
    if ($LASTEXITCODE -ne 0) { throw 'uv venv failed' }
}

uv pip install --system-certs --python .venv\Scripts\python.exe -e '.[dev,databento,gpu]'
if ($LASTEXITCODE -ne 0) { throw 'dependency install failed' }

.venv\Scripts\python.exe -m pytest
if ($LASTEXITCODE -ne 0) { throw 'tests failed' }
