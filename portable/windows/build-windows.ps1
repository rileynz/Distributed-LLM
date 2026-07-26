param(
    [string]$Version = "0.3.0",
    [switch]$SkipTests,
    [switch]$SkipBackend
)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$Dist = Join-Path $Project "dist\portable"
$Work = Join-Path $Project "build\portable-windows"
$Runtime = Join-Path $Project "portable\runtime\backend"
Set-Location $Project

function Find-PythonLauncher {
    if ($env:DLLM_PYTHON) { return @($env:DLLM_PYTHON) }
    if ($env:pythonLocation) {
        $SetupPython = Join-Path $env:pythonLocation "python.exe"
        if (Test-Path $SetupPython) { return @($SetupPython) }
    }
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) { return @($PythonCommand.Source) }
    if (Get-Command py -ErrorAction SilentlyContinue) { return @("py", "-3") }
    throw "Python 3.10 or newer was not found on this build machine."
}

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $Launcher = @(Find-PythonLauncher)
    $Executable = $Launcher[0]
    $Prefix = @()
    if ($Launcher.Length -gt 1) { $Prefix = $Launcher[1..($Launcher.Length - 1)] }
    & $Executable @Prefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

New-Item -ItemType Directory -Force -Path $Dist, $Runtime | Out-Null
Invoke-Python -m pip install --disable-pip-version-check -r requirements.txt "pyinstaller==6.14.2"
if (-not $SkipTests) {
    Invoke-Python -m unittest discover -v
}
if (-not $SkipBackend) {
    Invoke-Python portable/prepare_backend.py $Runtime --variant cpu
}

Invoke-Python -m PyInstaller --noconfirm --clean `
    --distpath $Dist `
    --workpath $Work `
    "portable/windows/PortableWorker.spec"

$Package = Join-Path $Dist "DistributedLLM-Portable-Worker"
$Executable = Join-Path $Package "DistributedLLM-Worker.exe"
if (-not (Test-Path $Executable)) {
    throw "Portable worker executable was not created."
}
& $Executable --version
if ($LASTEXITCODE -ne 0) {
    throw "Portable worker startup test failed."
}
Copy-Item "portable\Start-Portable-Worker.cmd" (Join-Path $Package "Start-Portable-Worker.cmd") -Force

$Archive = Join-Path $Dist "Distributed-LLM-Portable-Worker-Windows-x64-$Version.zip"
Compress-Archive -Path "$Package\*" -DestinationPath $Archive -Force
$Hash = Get-FileHash -Algorithm SHA256 $Archive
"$($Hash.Hash.ToLower())  $($Archive | Split-Path -Leaf)" |
    Set-Content -Encoding ascii (Join-Path $Dist "SHA256SUMS-windows-portable.txt")

Write-Host "Portable Windows worker ready: $Archive"

