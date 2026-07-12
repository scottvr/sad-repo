$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Push-Location $RepoRoot
try {
    foreach ($Seed in 0..9) {
        Invoke-PythonStep @(
            "scripts/evaluate_sequence.py",
            "--steps", "200",
            "--seed", "$Seed",
            "--out", "artifacts/sequence_seed_$Seed.json"
        )

        Invoke-PythonStep @(
            "scripts/run_controller.py",
            "--steps", "200",
            "--seed", "$Seed",
            "--anchor", "1.0",
            "--out", "artifacts/controller_anchor_seed_$Seed.json"
        )
    }
}
finally {
    Pop-Location
}
