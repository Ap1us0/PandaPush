param(
    [Parameter(Mandatory)]
    [ValidateSet(11, 29, 47)]
    [int]$Seed,
    [string]$OutputRoot = 'results/success_progress'
)

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    $python = '.\.venv\Scripts\python.exe'
    $checkpoint = "results/contact_safety/training/speed_limit/seed$Seed/final.zip"
    & $python -u -m panda_push.train --resume $checkpoint --continuation-seed (120000 + $Seed) `
        --steps 50000 --output "$OutputRoot/training/seed$Seed"
    if ($LASTEXITCODE -ne 0) { throw "Continuation failed for seed $Seed" }
}
finally {
    Pop-Location
}
