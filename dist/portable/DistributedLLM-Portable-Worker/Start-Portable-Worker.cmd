@echo off
setlocal
cd /d "%~dp0"
"DistributedLLM-Worker.exe" %*
if errorlevel 1 pause
