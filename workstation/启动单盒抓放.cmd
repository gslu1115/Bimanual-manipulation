@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -SinglePickPlace
if errorlevel 1 (
    echo Simulation failed. See failure.txt in the latest workstation outputs folder.
    pause
)
