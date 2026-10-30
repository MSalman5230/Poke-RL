$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$clientPath = Join-Path $projectRoot 'runtime_data/showdown_client'
$clientCommit = '02c67685cc09d0191a9be015de6db9dfaf5e4159'
if (-not (Test-Path -LiteralPath $clientPath)) {
    git init $clientPath
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the client checkout' }
    git -C $clientPath remote add origin https://github.com/smogon/pokemon-showdown-client.git
    if ($LASTEXITCODE -ne 0) { throw 'Could not configure the official client source' }
    git -C $clientPath fetch --depth 1 origin $clientCommit
    if ($LASTEXITCODE -ne 0) { throw 'Could not download the official Showdown client' }
    git -C $clientPath checkout --detach $clientCommit
    if ($LASTEXITCODE -ne 0) { throw 'Could not check out the pinned client' }
}
$currentCommit = git -C $clientPath rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $currentCommit -ne $clientCommit) {
    throw "The existing client is not the expected version: $clientPath"
}
Push-Location $clientPath
try {
    node build
    if ($LASTEXITCODE -ne 0) { throw 'Showdown client build failed' }
    node -e "require(process.argv[1]).prepareAssets(process.argv[2])" (Join-Path $projectRoot 'simulator/showdown_http.cjs') (Join-Path $clientPath 'play.pokemonshowdown.com')
    if ($LASTEXITCODE -ne 0) { throw 'Local battle graphics build failed' }
} finally { Pop-Location }
Write-Host 'Ready. Run .\.venv\Scripts\python.exe -m champions_rl.human_play from the project directory.'
