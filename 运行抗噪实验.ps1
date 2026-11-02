$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
$env:TEMP=Join-Path $PSScriptRoot 'work'
$env:TMP=$env:TEMP
$experimentPython='D:\software\miniconda3\envs\torch\python.exe'
if (-not (Test-Path -LiteralPath $experimentPython)) { throw 'Please set experimentPython to your CUDA Python path.' }
$runLog=Join-Path $PSScriptRoot ('results\reproduce-'+(Get-Date -Format 'yyyyMMdd-HHmmss')+'.log')
& $experimentPython -u (Join-Path $PSScriptRoot 'src\train_robust.py') 2>&1 | Tee-Object -FilePath $runLog
if ($LASTEXITCODE -ne 0) { throw 'Robust training failed.' }
& $experimentPython -u (Join-Path $PSScriptRoot 'src\verify_robust.py')
if ($LASTEXITCODE -ne 0) { throw 'Dual-condition acceptance failed.' }
& $experimentPython (Join-Path $PSScriptRoot 'src\infer_robust.py') --sigma 0.15
if ($LASTEXITCODE -ne 0) { throw 'Robust inference failed.' }
Write-Host 'GPU verification completed; every seed and ensemble passes both conditions.'
