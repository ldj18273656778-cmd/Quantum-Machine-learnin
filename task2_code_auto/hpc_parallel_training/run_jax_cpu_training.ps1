param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = 'Stop'

$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

$PythonExe = 'C:\ProgramData\anaconda3\envs\QML\python.exe'
$TrainBlock = Join-Path $PSScriptRoot 'train_block.py'

& $PythonExe $TrainBlock @RemainingArgs
exit $LASTEXITCODE
