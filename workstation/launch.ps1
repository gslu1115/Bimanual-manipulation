param(
    [string]$IsaacRoot = 'D:\isaacsim',
    [int]$Seed = 6,
    [switch]$Headless,
    [switch]$ConveyorTest,
    [switch]$SinglePickPlace,
    [switch]$DualHandover,
    [switch]$NoSensors,
    [switch]$RecordVideo,
    [string]$RobotUsd = ''
)
$ErrorActionPreference = 'Stop'
if (([int]$ConveyorTest.IsPresent + [int]$SinglePickPlace.IsPresent + [int]$DualHandover.IsPresent) -gt 1) { throw 'Choose exactly one task mode.' }
$pythonEntry = Join-Path $IsaacRoot 'python.bat'
if (-not (Test-Path -LiteralPath $pythonEntry)) { throw "Isaac Sim python.bat not found: $pythonEntry" }
$runArgs = @('-u', (Join-Path $PSScriptRoot 'run_scene.py'), '--seed', "$Seed")
$runName = 'seed_{0:D3}_{1}' -f $Seed, (Get-Date -Format 'yyyyMMdd_HHmmss')
if ($ConveyorTest) { $runName = 'conveyor_' + $runName; $runArgs += '--conveyor-test' }
if ($SinglePickPlace) { $runName = 'single_pnp_' + $runName; $runArgs += '--single-pnp' }
if ($DualHandover) { $runName = 'dual_handover_' + $runName; $runArgs += '--dual-handover' }
if ($NoSensors) { $runArgs += '--no-sensors' }
if ($RecordVideo) { $runArgs += '--record-video' }
$runArgs += @('--output', (Join-Path $PSScriptRoot "outputs\$runName"))
if ($Headless) { $runArgs += '--headless' } else { $runArgs += '--keep-open' }
if ($RobotUsd) { $runArgs += @('--robot-usd', $RobotUsd) }
& $pythonEntry @runArgs
exit $LASTEXITCODE
