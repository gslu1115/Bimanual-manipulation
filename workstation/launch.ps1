param(
    [string]$IsaacRoot = 'D:\isaacsim',
    [int]$Seed = 6,
    [switch]$Headless,
    [switch]$ConveyorTest,
    [switch]$SinglePickPlace,
    [string]$RobotUsd = ''
)
$ErrorActionPreference = 'Stop'
if ($ConveyorTest -and $SinglePickPlace) { throw 'Choose ConveyorTest or SinglePickPlace, not both.' }
$pythonEntry = Join-Path $IsaacRoot 'python.bat'
if (-not (Test-Path -LiteralPath $pythonEntry)) { throw "Isaac Sim python.bat not found: $pythonEntry" }
$runArgs = @('-u', (Join-Path $PSScriptRoot 'run_scene.py'), '--seed', "$Seed")
$runName = 'seed_{0:D3}_{1}' -f $Seed, (Get-Date -Format 'yyyyMMdd_HHmmss')
if ($ConveyorTest) { $runName = 'conveyor_' + $runName; $runArgs += '--conveyor-test' }
if ($SinglePickPlace) { $runName = 'single_pnp_' + $runName; $runArgs += '--single-pnp' }
$runArgs += @('--output', (Join-Path $PSScriptRoot "outputs\$runName"))
if ($Headless) { $runArgs += '--headless' } else { $runArgs += '--keep-open' }
if ($RobotUsd) { $runArgs += @('--robot-usd', $RobotUsd) }
& $pythonEntry @runArgs
exit $LASTEXITCODE
