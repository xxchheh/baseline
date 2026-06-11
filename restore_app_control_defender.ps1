$ErrorActionPreference = 'Continue'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath = Join-Path $PSScriptRoot "restore_app_control_defender_$stamp.log"
Start-Transcript -Path $logPath -Force | Out-Null

function Backup-RenameItem {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Suffix
    )

    if (Test-Path -LiteralPath $Path) {
        try {
            $item = Get-Item -LiteralPath $Path -Force
            try { $item.Attributes = 'Normal' } catch {}
            $parent = Split-Path -Path $Path -Parent
            $leaf = Split-Path -Path $Path -Leaf
            $destLeaf = "$leaf.$Suffix"
            Write-Host "Renaming $Path -> $destLeaf"
            Rename-Item -LiteralPath $Path -NewName $destLeaf -Force -ErrorAction Stop
        } catch {
            Write-Host "FAILED to rename $Path : $($_.Exception.Message)"
        }
    } else {
        Write-Host "Not present: $Path"
    }
}

try {
    $isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    Write-Host "Administrator: $isAdmin"
    if (-not $isAdmin) {
        Write-Host "This script must be run as Administrator."
        exit 1
    }

    $badPolicyId = '{a244370e-44c9-4c06-b551-f6016e563076}'
    $suffix = "quarantine_$stamp"

    Write-Host '=== Current App Control policies before ==='
    if (Test-Path "$env:windir\System32\CiTool.exe") {
        & "$env:windir\System32\CiTool.exe" -lp
    } else {
        Write-Host 'CiTool.exe not found.'
    }

    Write-Host '=== Remove suspicious App Control policy with CiTool ==='
    if (Test-Path "$env:windir\System32\CiTool.exe") {
        & "$env:windir\System32\CiTool.exe" --remove-policy $badPolicyId
        & "$env:windir\System32\CiTool.exe" --refresh
    }

    Write-Host '=== Quarantine single-policy WDAC file from OS volume ==='
    Backup-RenameItem -Path "$env:windir\System32\CodeIntegrity\SiPolicy.p7b" -Suffix $suffix

    Write-Host '=== Quarantine matching multiple-policy WDAC file from OS volume, if present ==='
    Backup-RenameItem -Path "$env:windir\System32\CodeIntegrity\CiPolicies\Active\$badPolicyId.cip" -Suffix $suffix

    Write-Host '=== Check EFI System Partition for App Control policy files ==='
    $used = (Get-PSDrive -PSProvider FileSystem).Name
    $driveLetter = @('Z','Y','X','W','V') | Where-Object { $used -notcontains $_ } | Select-Object -First 1
    if ($driveLetter) {
        $mountPoint = "$driveLetter`:"
        Write-Host "Mounting EFI System Partition at $mountPoint"
        & mountvol.exe $mountPoint /S
        Start-Sleep -Seconds 1
        if (Test-Path "$mountPoint\") {
            Backup-RenameItem -Path "$mountPoint\EFI\Microsoft\Boot\SiPolicy.p7b" -Suffix $suffix
            Backup-RenameItem -Path "$mountPoint\EFI\Microsoft\Boot\CiPolicies\Active\$badPolicyId.cip" -Suffix $suffix
            Write-Host 'EFI App Control files after quarantine:'
            Get-ChildItem -LiteralPath "$mountPoint\EFI\Microsoft\Boot" -Force -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'SiPolicy|CiPolicies|quarantine' } | Select-Object FullName,Length,CreationTime,LastWriteTime | Format-Table -AutoSize
        } else {
            Write-Host 'EFI mount did not appear.'
        }
        Write-Host "Unmounting EFI System Partition from $mountPoint"
        & mountvol.exe $mountPoint /D
    } else {
        Write-Host 'No spare drive letter available for EFI check.'
    }

    Write-Host '=== Remove malicious local policy artifacts ==='
    Backup-RenameItem -Path 'C:\ProgramData\ntuser.pol' -Suffix $suffix
    $gpPol = 'C:\Windows\System32\GroupPolicy\Machine\Registry.pol'
    if (Test-Path -LiteralPath $gpPol) {
        Write-Host "Backing up local machine Group Policy registry file: $gpPol"
        Copy-Item -LiteralPath $gpPol -Destination "$gpPol.backup_$stamp" -Force -ErrorAction SilentlyContinue
    }

    Write-Host '=== Remove Defender disable flags where permissions allow ==='
    $defenderKey = 'HKLM:\SOFTWARE\Microsoft\Windows Defender'
    if (Test-Path $defenderKey) {
        foreach ($name in @('DisableAntiSpyware', 'DisableAntiVirus')) {
            try {
                Write-Host "Setting $defenderKey :: $name = 0"
                Set-ItemProperty -Path $defenderKey -Name $name -Value 0 -Type DWord -ErrorAction Stop
            } catch {
                Write-Host "Could not set $name directly: $($_.Exception.Message)"
            }
        }
    }
    $policyDefenderKey = 'HKLM:\SOFTWARE\Policies\Microsoft\Windows Defender'
    if (Test-Path $policyDefenderKey) {
        foreach ($name in @('DisableAntiSpyware', 'DisableAntiVirus')) {
            if (Get-ItemProperty -Path $policyDefenderKey -Name $name -ErrorAction SilentlyContinue) {
                try {
                    Write-Host "Removing policy value $policyDefenderKey :: $name"
                    Remove-ItemProperty -Path $policyDefenderKey -Name $name -ErrorAction Stop
                } catch {
                    Write-Host "Could not remove policy value ${name}: $($_.Exception.Message)"
                }
            }
        }
    }

    Write-Host '=== Try to start Defender services ==='
    foreach ($svc in @('WinDefend', 'WdNisSvc', 'SecurityHealthService', 'wscsvc')) {
        try {
            Write-Host "Starting service $svc"
            Start-Service -Name $svc -ErrorAction Stop
        } catch {
            Write-Host "Could not start $svc now: $($_.Exception.Message)"
        }
    }
    try { Set-MpPreference -DisableRealtimeMonitoring $false -ErrorAction SilentlyContinue } catch {}
    try { Set-MpPreference -DisableIOAVProtection $false -ErrorAction SilentlyContinue } catch {}
    try { Set-MpPreference -DisableBehaviorMonitoring $false -ErrorAction SilentlyContinue } catch {}

    Write-Host '=== Current App Control policies after ==='
    if (Test-Path "$env:windir\System32\CiTool.exe") {
        & "$env:windir\System32\CiTool.exe" -lp
    }

    Write-Host '=== Defender/service status after ==='
    Get-Service -Name WinDefend,WdNisSvc,SecurityHealthService,wscsvc -ErrorAction SilentlyContinue | Select-Object Name,Status,StartType | Format-Table -AutoSize
    Get-MpComputerStatus -ErrorAction SilentlyContinue | Select-Object AMServiceEnabled,AntivirusEnabled,RealTimeProtectionEnabled,IoavProtectionEnabled,AntispywareEnabled,NISEnabled,AMServiceVersion,AntivirusSignatureLastUpdated | Format-List

    Write-Host 'A reboot is recommended now so Windows unloads the removed App Control policy.'
    Write-Host "Log written to: $logPath"
} finally {
    Stop-Transcript | Out-Null
}
