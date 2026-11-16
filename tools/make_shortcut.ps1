<#
.SYNOPSIS
  Create (or remove) ThaiASR shortcuts in the Start Menu and/or on the Desktop.

.EXAMPLE
  powershell -File tools\make_shortcut.ps1 -Exe dist\ThaiASR\ThaiASR.exe -What both
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [ValidateSet('both', 'startmenu', 'desktop', 'remove')][string]$What = 'both'
)

$ErrorActionPreference = 'Stop'
$exePath = (Resolve-Path -LiteralPath $Exe).Path
$workDir = Split-Path -Parent $exePath
$name = 'ThaiASR.lnk'

$startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
$desktop = [Environment]::GetFolderPath('Desktop')
$targets = @()
if ($What -eq 'both' -or $What -eq 'startmenu') { $targets += (Join-Path $startMenu $name) }
if ($What -eq 'both' -or $What -eq 'desktop') { $targets += (Join-Path $desktop $name) }

if ($What -eq 'remove') {
    foreach ($path in @((Join-Path $startMenu $name), (Join-Path $desktop $name))) {
        if (Test-Path -LiteralPath $path) {
            Remove-Item -LiteralPath $path -Force
            Write-Host "removed $path"
        }
    }
    exit 0
}

$shell = New-Object -ComObject WScript.Shell
foreach ($path in $targets) {
    $lnk = $shell.CreateShortcut($path)
    $lnk.TargetPath = $exePath
    $lnk.WorkingDirectory = $workDir
    $lnk.Description = 'ThaiASR - speech to text (Thai)'
    $lnk.IconLocation = "$exePath,0"
    $lnk.Save()
    Write-Host "created $path"
}
