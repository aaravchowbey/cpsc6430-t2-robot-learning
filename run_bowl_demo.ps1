param(
    [string]$Python,
    [string]$DataDir
)

$ErrorActionPreference = 'Stop'
if ($Python -and (Test-Path -LiteralPath $Python -PathType Leaf)) {
    $Python = (Resolve-Path -LiteralPath $Python).Path
}
if (-not $Python) {
    $venvPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        $venvPython = Join-Path $PSScriptRoot '.venv/bin/python'
    }
    $Python = if (Test-Path -LiteralPath $venvPython -PathType Leaf) { $venvPython } else { 'python' }
}
if (-not $DataDir) {
    $DataDir = Join-Path $PSScriptRoot 'data/libero_goal'
}
$DataDir = (Resolve-Path -LiteralPath $DataDir).Path
Push-Location -LiteralPath $PSScriptRoot
try {
    $runId = Get-Date -Format 'yyyyMMdd-HHmmss-ffff'
    $outputDir = Join-Path 'output/policies/execution_demo' $runId
    $results = Join-Path $outputDir 'bowl_results.csv'
    $videoDir = Join-Path $outputDir 'videos'
    $policyArgs = @(
        'run_policy.py'
        '--checkpoint', 'output/policies/dense.pt'
        '--data-dir', $DataDir
        '--task', '1'
        '--episode', 'demo_1'
        '--seed', '42'
        '--max-steps', '300'
        '--results', $results
        '--video-dir', $videoDir
    )

    if (-not $env:NUMBA_CACHE_DIR) {
        $env:NUMBA_CACHE_DIR = Join-Path ([System.IO.Path]::GetTempPath()) 'libero-numba'
    }
    $localConfig = Join-Path $PSScriptRoot '.libero'
    if (-not $env:LIBERO_CONFIG_PATH -and (Test-Path -LiteralPath (Join-Path $localConfig 'config.yaml'))) {
        $env:LIBERO_CONFIG_PATH = $localConfig
    }
    & $Python @policyArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Policy runner exited with code $LASTEXITCODE."
    }

    Write-Host "Results saved to $(Resolve-Path -LiteralPath $results)"
    Write-Host "Video saved under $(Resolve-Path -LiteralPath $videoDir)"
}
finally {
    Pop-Location
}
