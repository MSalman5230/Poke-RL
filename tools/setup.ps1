param([switch]$SkipTests)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

function Assert-Exit([string]$stage) {
    if ($LASTEXITCODE -ne 0) { throw "$stage failed with exit code $LASTEXITCODE" }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv and put it on PATH first: https://docs.astral.sh/uv/getting-started/installation/' }
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Install Node.js 22.18 or newer first: https://nodejs.org/' }

if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    uv venv --python 3.12 .venv
    Assert-Exit 'Virtual environment creation'
}
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
uv pip sync --python $projectPython requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
Assert-Exit 'Python dependency installation'
uv pip install --python $projectPython --no-deps -e .
Assert-Exit 'Editable project installation'
Push-Location -LiteralPath (Join-Path $projectRoot 'simulator')
try {
    npm ci --no-audit --no-fund
    Assert-Exit 'Simulator installation'
} finally { Pop-Location }

& $projectPython -m champions_rl status
Assert-Exit 'Environment check'
if (-not $SkipTests) {
    & $projectPython -m pytest -q
    Assert-Exit 'Tests'
}
Write-Output 'Ready. See README.md for offline play, training, and evaluation commands.'
