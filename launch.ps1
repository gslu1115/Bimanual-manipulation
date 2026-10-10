param(
    [ValidateSet('scene','single-pnp','dual-handover','conveyor-check','sensor-check','vision-check','visual-pnp','visual-sort','model-capture')]
    [string]$Mode = 'scene',
    [string]$IsaacRoot = '',
    [int]$Seed = 6,
    [switch]$Headless,
    [switch]$Livestream,
    [string]$StreamHost = '127.0.0.1',
    [switch]$NoSensors,
    [switch]$SaveImages,
    [switch]$CameraRecording,
    [switch]$RecordVideo,
    [string]$RobotUsd = '',
    [string]$Output = '',
    [ValidateSet('geometry','yoloe','yolo-seg')][string]$Segmenter = 'geometry',
    [ValidateSet('text','visual','trained')][string]$PromptMode = 'text',
    [string]$Weights = '',
    [string]$Reference = '',
    [ValidateSet('upright','side','inverted','separated','clutter')][string]$Fixture = 'upright',
    [ValidateSet('upright','side','inverted','separated-upright','separated','clutter')][string]$VisualFixture,
    [ValidateRange(10,50)][double]$SceneFocalLengthMm = 20,
    [ValidateRange(10,30)][double]$WristFocalLengthMm = 20,
    [ValidateRange(1,3)][int]$MaxItems = 3,
    [ValidateRange(0.0001,0.9999)][double]$Confidence = .4,
    [ValidateSet('front','rear','overhead','west')][string]$SceneView = 'front',
    [ValidateSet('default','cross-left','cross-left-offset')][string]$RightWristView = 'default',
    [ValidateRange(0.8,1.2)][double]$SceneLookHeightM = .8,
    [ValidateRange(1.1,2.5)][double]$SceneHeightM = 1.7,
    [ValidateRange(1,10)][double]$OpticalDepthMarginMm = 3,
    [switch]$DepthBackboard,
    [switch]$RobotStowHome,
    [switch]$PrepositionObserver
)
$ErrorActionPreference = 'Stop'
$onWindows = [Environment]::OSVersion.Platform -eq 'Win32NT'
if (-not $IsaacRoot) {
    if ($env:ISAAC_PATH) { $IsaacRoot = $env:ISAAC_PATH }
    elseif ($onWindows) { $IsaacRoot = 'D:\isaacsim' }
    else { $IsaacRoot = '/isaac-sim' }
}
if (-not $PSBoundParameters.ContainsKey('IsaacRoot') -and
    -not (Test-Path -LiteralPath (Join-Path $IsaacRoot 'python.bat')) -and
    $onWindows -and -not $env:ISAAC_PATH -and
    (Test-Path -LiteralPath 'D:\Issaccc\python.bat')) { $IsaacRoot = 'D:\Issaccc' }
$pythonEntry = Join-Path $IsaacRoot $(if ($onWindows) { 'python.bat' } else { 'python.sh' })
if (-not (Test-Path -LiteralPath $pythonEntry)) { throw "Isaac Sim Python launcher not found: $pythonEntry" }
$runArgs = @('-u', '-m', 'workstation', '--mode', $Mode, '--seed', "$Seed")
if ($Headless) { $runArgs += '--headless' } else { $runArgs += '--keep-open' }
if ($Livestream) { $runArgs += @('--livestream', '--stream-host', $StreamHost) }
if ($NoSensors) { $runArgs += '--no-sensors' }
if ($SaveImages) { $runArgs += '--save-images' }
if ($CameraRecording) { $runArgs += '--camera-recording' }
if ($RecordVideo) { $runArgs += '--record-video' }
if ($RobotUsd) { $runArgs += @('--robot-usd', $RobotUsd) }
if ($Output) { $runArgs += @('--output', $Output) }
$runArgs += @('--segmenter',$Segmenter,'--prompt-mode',$PromptMode,'--fixture',$Fixture)
if ($Weights) { $runArgs += @('--weights',$Weights) }
if ($Reference) { $runArgs += @('--reference',$Reference) }
if ($VisualFixture) { $runArgs += @('--visual-fixture',$VisualFixture) }
if ($PSBoundParameters.ContainsKey('SceneFocalLengthMm')) { $runArgs += @('--scene-focal-length-mm',"$SceneFocalLengthMm") }
if ($PSBoundParameters.ContainsKey('WristFocalLengthMm')) { $runArgs += @('--wrist-focal-length-mm',"$WristFocalLengthMm") }
if ($PSBoundParameters.ContainsKey('Confidence')) { $runArgs += @('--confidence',"$Confidence") }
if ($Mode -eq 'visual-sort') { $runArgs += @('--max-items',"$MaxItems") }
if ($SceneView -ne 'front') { $runArgs += @('--scene-view',$SceneView) }
if ($RightWristView -ne 'default') { $runArgs += @('--right-wrist-view',$RightWristView) }
if ($PSBoundParameters.ContainsKey('SceneLookHeightM')) { $runArgs += @('--scene-look-height-m',"$SceneLookHeightM") }
if ($PSBoundParameters.ContainsKey('SceneHeightM')) { $runArgs += @('--scene-height-m',"$SceneHeightM") }
if ($PSBoundParameters.ContainsKey('OpticalDepthMarginMm')) { $runArgs += @('--optical-depth-margin-mm',"$OpticalDepthMarginMm") }
if ($DepthBackboard) { $runArgs += '--depth-backboard' }
if ($RobotStowHome) { $runArgs += '--robot-stow-home' }
if ($PrepositionObserver) { $runArgs += '--preposition-observer' }
Push-Location -LiteralPath $PSScriptRoot
try {
    & $pythonEntry @runArgs
    $result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
