$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
.venv\Scripts\python.exe -m futures_intelligence.cli audit-sources

