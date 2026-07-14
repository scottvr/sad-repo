# Dims grid: is 96 a magic number, or just n_sites x k?
# PowerShell twin of run_dims.sh — see that file for the design rationale
# (five different allocations of the same 48-dim budget, plus an
# attention-only dims curve). All arms replay=1 --no-gates.
#
# Usage:
#   $env:PYTHON = ".\.venv\Scripts\python.exe"
#   .\scripts\run_dims.ps1 -Seeds @(0,1,2)

param(
    [int[]]$Seeds = @(0, 1, 2),
    [int]$Steps = 200,
    [string]$Model = "distilgpt2",
    [string]$OutRoot = "artifacts/sweeps/dims"
)

$ErrorActionPreference = "Stop"
$PythonBin = if ($env:PYTHON) { $env:PYTHON } else { "python" }

New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null

Write-Host "Writing dims-grid artifacts under $OutRoot"
Write-Host "Python: $PythonBin  Model: $Model  Steps: $Steps"
Write-Host "Seeds: $($Seeds -join ' ')"

function Run-Arm {
    param([string]$OutName, [int]$Seed, [string[]]$ExtraArgs)
    & $PythonBin scripts/run_controller.py `
        --model $Model --steps $Steps --seed $Seed `
        --no-gates --replay 1.0 @ExtraArgs `
        --out "$OutRoot/controller_${OutName}_seed_${Seed}.json"
    if ($LASTEXITCODE -ne 0) {
        throw "run_controller.py failed (arm=$OutName seed=$Seed)"
    }
}

foreach ($seed in $Seeds) {
    # dims curve on attention sites only: 12 / 24 / 48 / 96 dims
    foreach ($k in @(2, 4, 8, 16)) {
        Run-Arm "attn_k$k" $seed @("--sites", "attn", "--k", "$k")
    }
    # allocation controls at 48 dims (see run_dims.sh header)
    Run-Arm "mlp_k8" $seed @("--sites", "mlp")
    Run-Arm "early_k8" $seed @("--layers", "0-2")
    Run-Arm "late_k8" $seed @("--layers", "3-5")
}

Write-Host ""
Write-Host "Done. Aggregate with: python scripts/summarize_sweeps.py --stdout"
