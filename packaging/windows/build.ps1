param(
    [string]$Version = "0.3.0",
    [switch]$SkipTests,
    [switch]$InstallTools
)

$ErrorActionPreference = "Stop"
$Project = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$Dist = Join-Path $Project "dist\windows"
$Work = Join-Path $Project "build\windows"
Set-Location $Project

function Find-PythonLauncher {
    if ($env:DLLM_PYTHON) {
        return @($env:DLLM_PYTHON)
    }
    if ($env:pythonLocation) {
        $SetupPython = Join-Path $env:pythonLocation "python.exe"
        if (Test-Path $SetupPython) {
            return @($SetupPython)
        }
    }
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($PythonCommand) {
        return @($PythonCommand.Source)
    }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        return @("py", "-3")
    }
    throw "Python 3.10 or newer was not found."
}

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $Launcher = @(Find-PythonLauncher)
    $Executable = $Launcher[0]
    $Prefix = @()
    if ($Launcher.Length -gt 1) {
        $Prefix = $Launcher[1..($Launcher.Length - 1)]
    }
    & $Executable @Prefix @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE."
    }
}

Invoke-Python -m pip install --disable-pip-version-check -r requirements.txt "pyinstaller==6.14.2"
if (-not $SkipTests) {
    Invoke-Python -m unittest discover -v
}
Invoke-Python -m PyInstaller --noconfirm --clean `
    --distpath $Dist `
    --workpath $Work `
    "packaging/windows/DistributedLLM.spec"

$AppExecutable = Join-Path $Dist "DistributedLLM\DistributedLLM.exe"
if (-not (Test-Path $AppExecutable)) {
    throw "PyInstaller did not create $AppExecutable."
}

& $AppExecutable --version
if ($LASTEXITCODE -ne 0) {
    throw "The packaged application did not pass its startup test."
}

$CompilerCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)
$Compiler = $CompilerCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $Compiler -and $InstallTools) {
    if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
        throw "Inno Setup is missing and Chocolatey is not available to install it."
    }
    choco install innosetup --yes --no-progress
    if ($LASTEXITCODE -ne 0) {
        throw "Chocolatey could not install Inno Setup."
    }
    $Compiler = $CompilerCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
}
if (-not $Compiler) {
    throw "Inno Setup 6 was not found. Install it from https://jrsoftware.org/isdl.php and run this script again."
}

& $Compiler "/DMyAppVersion=$Version" "packaging/windows/installer.iss"
if ($LASTEXITCODE -ne 0) {
    throw "Inno Setup failed with exit code $LASTEXITCODE."
}

$Installer = Join-Path $Dist "Distributed-LLM-Setup-$Version.exe"
if (-not (Test-Path $Installer)) {
    throw "The installer was not created at $Installer."
}

$Hash = Get-FileHash -Algorithm SHA256 $Installer
"$($Hash.Hash.ToLower())  $($Installer | Split-Path -Leaf)" |
    Set-Content -Encoding ascii (Join-Path $Dist "SHA256SUMS.txt")

Write-Host ""
Write-Host "Windows installer ready:"
Write-Host "  $Installer"

