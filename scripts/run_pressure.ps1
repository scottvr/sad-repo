# Pressure grid: stress-test the replay retention result before trusting it.
# PowerShell twin of run_pressure.sh — see that file (or
# docs/roadmap_v0.2.md) for what each arm is for.
#
# Usage:
#   $env:PYTHON = ".\.venv\Scripts\python.exe"
#   .\scripts\run_pressure.ps1 -Seeds @(0,1,2)

param(
    [int[]]$Seeds = @(0, 1, 2, 3, 4),
    [int]$Steps = 200,
    [string]$Model = "distilgpt2",
    [string]$OutRoot = "artifacts/sweeps/pressure"
)

$ErrorActionPreference = "Stop"
$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

Write-Host "Writing pressure-grid artifacts under $OutRoot"
Write-Host "Python: $PythonBin  Model: $Model  Steps: $Steps"
Write-Host "Seeds: $($Seeds -join ' ')"

function Run-Arm {
    param([string]$OutName, [int]$Seed, [string[]]$ExtraArgs)
    & $PythonBin scripts/run_controller.py `
        --model $Model --steps $Steps --seed $Seed --no-gates @ExtraArgs `
        --out "$OutRoot/controller_${OutName}_seed_${Seed}.json"
    if ($LASTEXITCODE -ne 0) {
        throw "run_controller.py failed (arm=$OutName seed=$Seed)"
    }
}

foreach ($seed in $Seeds) {
    # 1) replay-fraction sweep (default data)
    foreach ($frac in @(0.125, 0.25, 0.5)) {
        $ftag = "$frac".Replace(".", "p")
        Run-Arm "frac_$ftag" $seed @("--replay", "1.0",
                                     "--replay-fraction", "$frac")
    }
    # 2) bigger family, ceiling removed: baseline + replay
    Run-Arm "big_replay_0" $seed @("--facts-per-task", "8", "--wide-labels")
    Run-Arm "big_replay_1p0" $seed @("--facts-per-task", "8", "--wide-labels",
                                     "--replay", "1.0")
    # 3) conflicting facts across domains: baseline + replay
    Run-Arm "conflict_replay_0" $seed @("--overlap-words", "2")
    Run-Arm "conflict_replay_1p0" $seed @("--overlap-words", "2",
                                          "--replay", "1.0")
    # 4) capacity: k x replay (k=8 covered by the retention grid)
    foreach ($k in @(2, 4, 16)) {
        Run-Arm "cap_k$k" $seed @("--k", "$k", "--replay", "1.0")
    }
}

Write-Host ""
Write-Host "Done. Aggregate with: python scripts/summarize_sweeps.py --stdout"
