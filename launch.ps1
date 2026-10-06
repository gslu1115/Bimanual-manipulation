param(
    [ValidateSet('scene','single-pnp','dual-handover','conveyor-check','sensor-check')]
    [string]$Mode = 'scene',
    [string]$IsaacRoot = 'D:\isaacsim',
    [int]$Seed = 6,
    [switch]$Headless,
    [switch]$NoSensors,
    [switch]$SaveImages,
    [switch]$CameraRecording,
    [switch]$RecordVideo,
    [string]$RobotUsd = '',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
if (-not $PSBoundParameters.ContainsKey('IsaacRoot') -and
    -not (Test-Path -LiteralPath (Join-Path $IsaacRoot 'python.bat')) -and
    (Test-Path -LiteralPath 'D:\Issaccc\python.bat')) { $IsaacRoot = 'D:\Issaccc' }
$pythonEntry = Join-Path $IsaacRoot 'python.bat'
if (-not (Test-Path -LiteralPath $pythonEntry)) { throw "Isaac Sim python.bat not found: $pythonEntry" }
$runArgs = @('-u', '-m', 'workstation', '--mode', $Mode, '--seed', "$Seed")
if ($Headless) { $runArgs += '--headless' } else { $runArgs += '--keep-open' }
if ($NoSensors) { $runArgs += '--no-sensors' }
if ($SaveImages) { $runArgs += '--save-images' }
if ($CameraRecording) { $runArgs += '--camera-recording' }
if ($RecordVideo) { $runArgs += '--record-video' }
if ($RobotUsd) { $runArgs += @('--robot-usd', $RobotUsd) }
if ($Output) { $runArgs += @('--output', $Output) }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonEntry @runArgs
    $result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
