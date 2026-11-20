param(
    [string]$Python,
    [string]$DataDir,
    [string]$OutputDir
)

$ErrorActionPreference = 'Stop'
$evaluator = Join-Path $PSScriptRoot 'evaluate_heldout.py'
if (-not (Test-Path -LiteralPath $evaluator -PathType Leaf)) {
    throw 'evaluate_heldout.py is missing. Update your checkout to the team''s latest main branch.'
}

if ($Python -and (Test-Path -LiteralPath $Python -PathType Leaf)) {
    $Python = (Resolve-Path -LiteralPath $Python).Path
}
if (-not $Python) {
    $venvPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    $Python = if (Test-Path -LiteralPath $venvPython -PathType Leaf) { $venvPython } else { 'python' }
}
if (-not $DataDir) {
    $DataDir = Join-Path $PSScriptRoot 'data/libero_goal'
}
if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    throw "Dataset folder not found: $DataDir. Pass -DataDir with your LIBERO-Goal HDF5 folder."
}
$DataDir = (Resolve-Path -LiteralPath $DataDir).Path
if ($OutputDir) {
    $OutputDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDir)
}

Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not $env:NUMBA_CACHE_DIR) {
        $env:NUMBA_CACHE_DIR = Join-Path ([System.IO.Path]::GetTempPath()) 'libero-numba'
    }
    $localConfig = Join-Path $PSScriptRoot '.libero'
    if (-not $env:LIBERO_CONFIG_PATH -and (Test-Path -LiteralPath (Join-Path $localConfig 'config.yaml'))) {
        $env:LIBERO_CONFIG_PATH = $localConfig
    }

    if (-not $OutputDir) {
        $OutputDir = Join-Path 'output/policies/execution_demo' (Get-Date -Format 'yyyyMMdd-HHmmss-ffff')
    }
    Write-Host "Output folder: $OutputDir"
    $tasks = @(
        @{ Name = 'drawer'; Id = 0; FirstInit = 37 }
        @{ Name = 'bowl'; Id = 1; FirstInit = 30 }
    )

    foreach ($task in $tasks) {
        $taskOutput = Join-Path $OutputDir $task.Name
        $results = Join-Path $taskOutput 'rollouts.csv'
        $completed = @{}
        if (Test-Path -LiteralPath $results) {
            foreach ($row in (Import-Csv -LiteralPath $results)) {
                if ($row.video -and (Test-Path -LiteralPath $row.video -PathType Leaf)) {
                    $completed["$($row.variant)/$($row.init_index)"] = $true
                } else {
                    throw "A completed rollout in $results is missing its video. Use a fresh output folder."
                }
            }
        }

        foreach ($index in $task.FirstInit..($task.FirstInit + 2)) {
            foreach ($model in @('vision', 'dense', 'grid')) {
                if ($completed.ContainsKey("$model/$index")) {
                    continue
                }
                Write-Host "Recording $($task.Name) | $model | start $index..."
                # Give each rollout a fresh simulator process and native rendering context.
                & $Python $evaluator --data-dir $DataDir --tasks $task.Id --first-init $task.FirstInit --n 3 --videos all --out-dir $taskOutput --only-rollout $model $index
                if ($LASTEXITCODE -ne 0) {
                    throw "Evaluation failed for $($task.Name), $model, start $index (exit $LASTEXITCODE). Resume with -OutputDir '$OutputDir'."
                }
            }
        }
        $clips = @(Get-ChildItem -LiteralPath (Join-Path $taskOutput 'videos') -Filter '*.mp4' -File)
        if ($clips.Count -ne 9) {
            throw "Expected 9 clips for $($task.Name), found $($clips.Count). Check $taskOutput."
        }
    }

    Write-Host "Saved 18 clips and two rollout CSVs under $(Resolve-Path -LiteralPath $OutputDir)"
}
finally {
    Pop-Location
}
