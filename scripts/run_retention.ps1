# Retention-mechanism grid: {replay off/on} x {soft ortho / hard projection},
# controller only. Artifacts land under artifacts/sweeps/retention/ so
# summarize_sweeps.py picks them up alongside the earlier sweeps.
#
# Usage:
#   $env:PYTHON = ".\.venv\Scripts\python.exe"
#   .\scripts\run_retention.ps1 -Seeds @(0,1,2) -ReplayValues @(0,0.3,1.0)

param(
    [int[]]$Seeds = @(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    [double[]]$ReplayValues = @(0, 1.0),
    [int]$Steps = 200,
    [string]$OutRoot = "artifacts/sweeps/retention"
)

$ErrorActionPreference = "Stop"
$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

Write-Host "Writing retention-grid artifacts under $OutRoot"
Write-Host "Python: $PythonBin"
Write-Host "Steps: $Steps"
Write-Host "Seeds: $($Seeds -join ' ')"
Write-Host "Replay weights: $($ReplayValues -join ' ')"

foreach ($seed in $Seeds) {
    foreach ($replay in $ReplayValues) {
        $rtag = "$replay".Replace(".", "p")
        # Whole grid runs --no-gates: gates were inert in earlier sweeps and
        # hard projection is only exact without them (single-variable arms).
        & $PythonBin scripts/run_controller.py `
            --steps $Steps --seed $seed --replay $replay --no-gates `
            --out "$OutRoot/controller_replay_${rtag}_seed_${seed}.json"
        if ($LASTEXITCODE -ne 0) { throw "run_controller.py failed (seed=$seed replay=$replay)" }
        & $PythonBin scripts/run_controller.py `
            --steps $Steps --seed $seed --replay $replay --no-gates --hard-ortho `
            --out "$OutRoot/controller_replay_${rtag}_hard_seed_${seed}.json"
        if ($LASTEXITCODE -ne 0) { throw "run_controller.py failed (seed=$seed replay=$replay hard)" }
    }
}

Write-Host ""
Write-Host "Done. Aggregate with: python scripts/summarize_sweeps.py --stdout"
