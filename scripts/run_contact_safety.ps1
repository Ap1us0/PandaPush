param(
    [Parameter(Mandatory)]
    [ValidateRange(0, 2)]
    [int]$Worker,
    [string]$OutputRoot = 'results/contact_safety'
)

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    $python = '.\.venv\Scripts\python.exe'
    $arms = @('control', 'speed_limit', 'force_penalty')
    $seeds = @(11, 29, 47)
    for ($index = 0; $index -lt $seeds.Count; $index++) {
        $arm = $arms[($Worker + $index) % $arms.Count]
        $seed = $seeds[$index]
        $run = "$OutputRoot/training/$arm/seed$seed"
        $test = "$OutputRoot/test/$arm/seed$seed"
        Write-Output "Worker $Worker runs $arm seed $seed"
        if (-not (Test-Path -LiteralPath "$run/training_summary.json")) {
            $options = switch ($arm) {
                'speed_limit' { '--contact-speed-mps'; '0.30' }
                'force_penalty' { '--force-penalty-weight'; '0.12' }
            }
            & $python -m panda_push.train --config configs/contact_safety.json --seed $seed --steps 50000 --output $run @options
            if ($LASTEXITCODE -ne 0) { throw "Training failed for $arm seed $seed" }
        }
        if (-not (Test-Path -LiteralPath "$test/complete.json")) {
            & $python -m panda_push.contact_study --checkpoint "$run/final.zip" --output $test --seed 97000 --episodes 100 --profiles nominal size_ood friction_ood ood
            if ($LASTEXITCODE -ne 0) { throw "Held-out evaluation failed for $arm seed $seed" }
        }
    }
}
finally {
    Pop-Location
}

