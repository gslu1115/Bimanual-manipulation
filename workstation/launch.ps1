param(
    [string]$IsaacRoot = 'D:\isaacsim',
    [int]$Seed = 6,
    [switch]$Headless,
    [switch]$ConveyorTest,
    [switch]$SinglePickPlace,
    [switch]$DualHandover,
    [switch]$NoSensors,
    [switch]$RecordVideo,
    [switch]$NoSaveImages,
    [switch]$CameraRecording,
    [switch]$CameraCheck,
    [string]$RobotUsd = ''
)
$ErrorActionPreference = 'Stop'
if (-not $PSBoundParameters.ContainsKey('IsaacRoot') -and -not (Test-Path -LiteralPath $IsaacRoot) -and (Test-Path -LiteralPath 'D:\Issaccc\python.bat')) { $IsaacRoot = 'D:\Issaccc' }
if (([int]$ConveyorTest.IsPresent + [int]$SinglePickPlace.IsPresent + [int]$DualHandover.IsPresent) -gt 1) { throw 'Choose exactly one task mode.' }
$pythonEntry = Join-Path $IsaacRoot 'python.bat'
if (-not (Test-Path -LiteralPath $pythonEntry)) { throw "Isaac Sim python.bat not found: $pythonEntry" }
$entry = if ($CameraCheck) { 'validate_cameras.py' } else { 'run_scene.py' }
if ($CameraCheck -and ($ConveyorTest -or $DualHandover -or $NoSensors -or $RecordVideo)) { throw 'CameraCheck supports the scene or SinglePickPlace, with sensors enabled.' }
$runArgs = @('-u', (Join-Path $PSScriptRoot $entry), '--seed', "$Seed")
$runName = 'seed_{0:D3}_{1}' -f $Seed, (Get-Date -Format 'yyyyMMdd_HHmmss')
if ($ConveyorTest) { $runName = 'conveyor_' + $runName; $runArgs += '--conveyor-test' }
if ($SinglePickPlace) { $runName = 'single_pnp_' + $runName; $runArgs += '--single-pnp' }
if ($DualHandover) { $runName = 'dual_handover_' + $runName; $runArgs += '--dual-handover' }
if ($NoSensors) { $runArgs += '--no-sensors' }
if ($RecordVideo) { $runArgs += '--record-video' }
if ($NoSaveImages) { $runArgs += '--no-save-images' }
if ($CameraRecording) { $runArgs += '--camera-recording' }
if ($CameraCheck) { $runName = 'camera_check_' + $runName }
$runArgs += @('--output', (Join-Path $PSScriptRoot "outputs\$runName"))
if ($Headless) { $runArgs += '--headless' } else { $runArgs += '--keep-open' }
if ($RobotUsd) { $runArgs += @('--robot-usd', $RobotUsd) }
& $pythonEntry @runArgs
exit $LASTEXITCODE
