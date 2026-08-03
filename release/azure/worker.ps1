param([switch]$Register)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$WorkRoot = 'C:\dl4j-release'
$BootstrapRoot = Join-Path $WorkRoot 'bootstrap'
$PersistedWorker = Join-Path $BootstrapRoot 'worker.ps1'
$WorkerStartedMarker = Join-Path $BootstrapRoot 'worker-started.txt'
$TaskName = 'DL4JReleaseWorker'
$LaneLog = Join-Path $WorkRoot 'lane.log'

if ($Register) {
  try {
    New-Item -ItemType Directory -Force -Path $WorkRoot,$BootstrapRoot | Out-Null
    Copy-Item -LiteralPath $PSCommandPath -Destination $PersistedWorker -Force
    Remove-Item -LiteralPath $WorkerStartedMarker -Force -ErrorAction SilentlyContinue
    $TaskCommand = "& '$PersistedWorker' *>> '$LaneLog'"
    $TaskArguments = '-NoLogo -NonInteractive -ExecutionPolicy Bypass -Command "' + $TaskCommand + '"'
    $Action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $TaskArguments
    $Trigger = New-ScheduledTaskTrigger -AtStartup
    $Principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    $Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
    Start-ScheduledTask -TaskName $TaskName
    $Deadline = [DateTime]::UtcNow.AddSeconds(90)
    while (-not (Test-Path -LiteralPath $WorkerStartedMarker)) {
      if ([DateTime]::UtcNow -ge $Deadline) {
        $Info = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
        throw "Scheduled worker did not start within 90 seconds (lastResult=$($Info.LastTaskResult))"
      }
      Start-Sleep -Seconds 2
    }
  }
  catch {
    Write-Output ($_ | Out-String)
    try {
      $global:LASTEXITCODE = $null
      & shutdown.exe /s /t 0 /f
      if ($null -ne $global:LASTEXITCODE -and $global:LASTEXITCODE -ne 0) {
        throw "shutdown.exe failed with exit code $global:LASTEXITCODE"
      }
    }
    catch {
      Write-Warning "shutdown.exe was unsuccessful after worker registration failure; using Stop-Computer: $($_.Exception.Message)"
      try { Stop-Computer -Force } catch { Write-Warning "Unable to shut down after worker registration failure: $($_.Exception.Message)" }
    }
    exit 1
  }
  exit 0
}

$ConfigB64 = '__DL4J_WORKER_CONFIG_B64__'
$BuildDriverB64 = '__DL4J_BUILD_DRIVER_B64__'
$CloudIoB64 = '__DL4J_CLOUD_IO_B64__'
$ToolchainRoot = Join-Path $WorkRoot 'toolchains'
$SourceRoot = Join-Path $WorkRoot 'sources'
$OutputRoot = Join-Path $WorkRoot 'outputs'
$MavenRepoRoot = Join-Path $WorkRoot 'm2'
$LaneForwarderStop = Join-Path $WorkRoot 'lane-forwarder.stop'
$LaneForwarderError = Join-Path $WorkRoot 'lane-forwarder.err'
$KillRequestedFile = Join-Path $WorkRoot 'kill-requested'
$WatchdogStopFile = Join-Path $WorkRoot 'kill-watchdog.stop'
$WatchdogCloudPidFile = Join-Path $WorkRoot 'kill-watchdog-cloud.pid'
$BuildPidFile = Join-Path $WorkRoot 'build.pid'
$ConfigFile = Join-Path $BootstrapRoot 'worker.json'
$BuildDriver = Join-Path $BootstrapRoot 'build-platform.py'
$CloudIo = Join-Path $BootstrapRoot 'cloud-io.py'
$AttemptFile = Join-Path $BootstrapRoot 'worker-attempt.txt'
$env:CARGO_HOME = Join-Path $ToolchainRoot 'cargo'
$env:RUSTUP_HOME = Join-Path $ToolchainRoot 'rustup'
$env:TEMP = Join-Path $WorkRoot 'tmp'
$env:TMP = $env:TEMP
$env:PATH = "$($env:CARGO_HOME)\bin;$env:PATH"
$Attempt = 1
$Config = $null
$Shards = @()
$ExitCode = 1
$KillWatchJob = $null
$LogForwarderProcess = $null
$LaneForwarderProcess = $null
$TranscriptStarted = $false
$CurrentActive = $false
$CurrentFinalized = $true
$script:Shard = $null
$script:SourceDir = $null
$script:OutputDir = $null
$script:BuildLog = $null
$script:MatrixLog = $null
$script:MatrixError = $null
$script:ShardConfigFile = $null
$script:ShardMavenRepo = $null
$script:ObjectPrefix = $null
$script:LogForwarderStop = $null
$script:LogForwarderError = $null
$script:PythonExe = $null
$script:WindowsTarExe = Join-Path $env:SystemRoot 'System32\tar.exe'
$script:MatrixLogOffsets = @{}

function Write-BuildContent([string]$Text) {
  if (-not $Text) { return }
  Write-Output $Text
  if ($script:BuildLog -and -not $script:TranscriptStarted) {
    Add-Content -Path $script:BuildLog -Value $Text
  }
}

function Write-Phase([string]$Phase, [string]$Status, [string]$Detail = '') {
  $Message = "[dl4j-phase] timestamp=$([DateTimeOffset]::UtcNow.ToString('o')) phase=$Phase status=$Status $Detail"
  Write-BuildContent $Message
}

function Invoke-NativeChecked {
  param(
    [Parameter(Mandatory=$true)][scriptblock]$Command,
    [Parameter(Mandatory=$true)][string]$Description,
    [int[]]$SuccessCodes = @(0)
  )
  $PreviousPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'Continue'
    # Native invocations update the automatic global variable; an unscoped
    # assignment here would create a function-local shadow that never changes.
    $global:LASTEXITCODE = $null
    & $Command
    $Code = $global:LASTEXITCODE
  }
  finally {
    $ErrorActionPreference = $PreviousPreference
  }
  if ($null -eq $Code) {
    throw "$Description did not report an exit code"
  }
  if ($SuccessCodes -notcontains [int]$Code) {
    throw "$Description failed with exit code $Code"
  }
}

function Invoke-KillSwitchProbe {
  & $script:PythonExe $CloudIo kill-enabled --bucket $Config.killSwitchBucket --object $Config.killSwitchObject --controller-epoch $Config.controllerEpoch --client-id $Config.managedIdentityClientId *> $null
  return $LASTEXITCODE
}

function Test-KillSwitch {
  $State = Invoke-KillSwitchProbe
  if ($State -eq 1) { return $false }
  if ($State -ne 0) { Write-Warning "Global kill switch is unreadable (cloud-io exit $State); failing closed" }
  return $true
}

function Wait-ForCloudAccess {
  $Deadline = [DateTime]::UtcNow.AddMinutes(15)
  $ProbeAttempt = 0
  while ($true) {
    $ProbeAttempt += 1
    $State = Invoke-KillSwitchProbe
    if ($State -eq 0) { throw 'Global kill switch is enabled' }
    if ($State -eq 1) {
      Write-Output "[dl4j-phase] timestamp=$([DateTimeOffset]::UtcNow.ToString('o')) phase=azure-blob-auth status=ready attempt=$ProbeAttempt"
      return
    }
    if ([DateTime]::UtcNow -ge $Deadline) {
      throw "Azure Blob access remained unavailable after 15 minutes (cloud-io exit $State)"
    }
    Write-Output "[dl4j-phase] timestamp=$([DateTimeOffset]::UtcNow.ToString('o')) phase=azure-blob-auth status=waiting attempt=$ProbeAttempt exit=$State"
    Start-Sleep -Seconds 15
  }
}

function Assert-NotKilled {
  if (Test-Path -LiteralPath $KillRequestedFile) {
    $Reason = (Get-Content -Raw $KillRequestedFile).Trim()
    if (-not $Reason) { $Reason = 'requested' }
    throw "Global kill switch stopped the lane ($Reason)"
  }
}

function Test-RemoteShardSuccess($Candidate) {
  $SafeId = $Candidate.id -replace '[^A-Za-z0-9._-]', '-'
  $StatusFile = Join-Path $BootstrapRoot "remote-$SafeId.json"
  Remove-Item -LiteralPath $StatusFile -Force -ErrorAction SilentlyContinue
  & $script:PythonExe $CloudIo download --bucket $Config.bucket --object "$($Config.artifactPrefix)/$($Config.runId)/$($Candidate.id)/status.json" --file $StatusFile --client-id $Config.managedIdentityClientId *> $null
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $StatusFile)) { return $false }
  try {
    $Status = Get-Content -Raw $StatusFile | ConvertFrom-Json
    $ExpectedVariants = @($Candidate.build.variants | ForEach-Object { $_.name }) | ConvertTo-Json -Compress
    $ActualVariants = @($Status.variants) | ConvertTo-Json -Compress
    return (
      $Status.shard -eq $Candidate.id -and
      $Status.runId -eq $Config.runId -and
      $Status.controllerEpoch -eq $Config.controllerEpoch -and
      $Status.repository -eq $Config.repository -and
      $Status.commit -eq $Config.commit -and
      $Status.releaseVersion -eq $Config.releaseVersion -and
      $Status.snapshotVersion -eq $Config.snapshotVersion -and
      $Status.contractDigest -eq $Candidate.contractDigest -and
      $ActualVariants -eq $ExpectedVariants -and
      [int]$Status.exitCode -eq 0
    )
  }
  catch {
    return $false
  }
}

function Set-ShardContext($NextShard) {
  $script:Shard = $NextShard
  $SafeId = $Shard.id -replace '[^A-Za-z0-9._-]', '-'
  $script:SourceDir = Join-Path $SourceRoot $SafeId
  $script:OutputDir = Join-Path $OutputRoot $SafeId
  $script:BuildLog = Join-Path $OutputDir 'build.log'
  $script:MatrixLog = Join-Path $OutputDir 'matrix-build.log'
  $script:MatrixError = Join-Path $OutputDir 'matrix-build.err'
  $script:ShardConfigFile = Join-Path $BootstrapRoot "$SafeId.json"
  $script:ShardMavenRepo = Join-Path $MavenRepoRoot $SafeId
  $script:ObjectPrefix = "$($Config.artifactPrefix)/$($Config.runId)/$($Shard.id)"
  $script:LogForwarderStop = Join-Path $OutputDir 'log-forwarder.stop'
  $script:LogForwarderError = Join-Path $OutputDir 'log-forwarder.err'
  Remove-Item -LiteralPath $SourceDir -Recurse -Force -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $OutputDir,$ShardMavenRepo | Out-Null
  Get-ChildItem -LiteralPath $OutputDir -Force -ErrorAction SilentlyContinue |
    Where-Object Name -ne 'build.log' |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  if ($Attempt -gt 1 -and (Test-Path $BuildLog)) {
    Add-Content -Path $BuildLog -Value "[dl4j-phase] timestamp=$([DateTimeOffset]::UtcNow.ToString('o')) phase=worker-restart status=started attempt=$Attempt shard=$($Shard.id)"
  }
  $ShardConfig = $Config | ConvertTo-Json -Depth 100 | ConvertFrom-Json
  $ShardConfig | Add-Member -NotePropertyName shard -NotePropertyValue $Shard -Force
  $ShardConfig.PSObject.Properties.Remove('shards')
  $ShardConfig | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $ShardConfigFile
  $script:MatrixLogOffsets = @{}
}

function Copy-NewLogContent([string]$Path) {
  if (-not (Test-Path $Path)) { return }
  if (-not $script:MatrixLogOffsets.ContainsKey($Path)) {
    $script:MatrixLogOffsets[$Path] = [int64]0
  }
  $Stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
  try {
    $Position = [int64]$script:MatrixLogOffsets[$Path]
    if ($Stream.Length -lt $Position) { $Position = 0 }
    [void]$Stream.Seek($Position, [IO.SeekOrigin]::Begin)
    $Reader = [IO.StreamReader]::new($Stream, [Text.Encoding]::UTF8, $true, 4096, $true)
    try {
      $Text = $Reader.ReadToEnd()
      $script:MatrixLogOffsets[$Path] = $Stream.Position
    }
    finally {
      $Reader.Dispose()
    }
    if ($Text) { Write-BuildContent $Text }
  }
  finally {
    $Stream.Dispose()
  }
}

function Start-ShardLogging {
  Remove-Item -LiteralPath $LogForwarderStop -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $LogForwarderError,"$LogForwarderError.stderr" -Force -ErrorAction SilentlyContinue
  Start-Transcript -Path $BuildLog -Append -Force | Out-Null
  $script:TranscriptStarted = $true
  $LogArguments = @(
    $CloudIo, 'forward', '--bucket', $Config.bucket,
    '--object', "$ObjectPrefix/live.log", '--file', $BuildLog,
    '--stop-file', $LogForwarderStop, '--client-id', $Config.managedIdentityClientId
  )
  $script:LogForwarderProcess = Start-Process $script:PythonExe -ArgumentList $LogArguments -RedirectStandardOutput $LogForwarderError -RedirectStandardError "$LogForwarderError.stderr" -PassThru -NoNewWindow
  Write-Phase 'azure-blob-log-forwarder' 'started' "lane=$($Config.laneId) shard=$($Shard.id) pid=$($LogForwarderProcess.Id)"
}

function Start-LaneLogging {
  if ($LaneForwarderProcess) { return }
  Remove-Item -LiteralPath $LaneForwarderStop -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $LaneForwarderError,"$LaneForwarderError.stderr" -Force -ErrorAction SilentlyContinue
  if (-not (Test-Path -LiteralPath $LaneLog)) {
    New-Item -ItemType File -Force -Path $LaneLog | Out-Null
  }
  $LaneArguments = @(
    $CloudIo, 'forward', '--bucket', $Config.bucket,
    '--object', "$($Config.artifactPrefix)/$($Config.runId)/lanes/$($Config.laneId)/live.log",
    '--file', $LaneLog, '--stop-file', $LaneForwarderStop,
    '--client-id', $Config.managedIdentityClientId
  )
  $script:LaneForwarderProcess = Start-Process $script:PythonExe -ArgumentList $LaneArguments -RedirectStandardOutput $LaneForwarderError -RedirectStandardError "$LaneForwarderError.stderr" -PassThru -NoNewWindow
  Write-Phase 'azure-blob-lane-forwarder' 'started' "lane=$($Config.laneId) pid=$($LaneForwarderProcess.Id)"
}

function Stop-LaneLogging {
  if (-not $LaneForwarderProcess) { return }
  New-Item -ItemType File -Force -Path $LaneForwarderStop | Out-Null
  Wait-Process -Id $LaneForwarderProcess.Id -Timeout 40 -ErrorAction SilentlyContinue
  $LaneForwarderProcess.Refresh()
  if (-not $LaneForwarderProcess.HasExited) {
    Stop-Process -Id $LaneForwarderProcess.Id -Force -ErrorAction SilentlyContinue
  }
  $script:LaneForwarderProcess = $null
}

function Stop-ShardLogging {
  if ($TranscriptStarted) {
    Stop-Transcript | Out-Null
    $script:TranscriptStarted = $false
  }
  if ($LogForwarderProcess) {
    New-Item -ItemType File -Force -Path $LogForwarderStop | Out-Null
    Wait-Process -Id $LogForwarderProcess.Id -Timeout 40 -ErrorAction SilentlyContinue
    $LogForwarderProcess.Refresh()
    if (-not $LogForwarderProcess.HasExited) {
      Stop-Process -Id $LogForwarderProcess.Id -Force -ErrorAction SilentlyContinue
    }
    $script:LogForwarderProcess = $null
  }
  if (Test-Path $LogForwarderError) { Get-Content $LogForwarderError | Add-Content $BuildLog }
  if (Test-Path "$LogForwarderError.stderr") { Get-Content "$LogForwarderError.stderr" | Add-Content $BuildLog }
}

function Upload-IfPresent([string]$Path, [string]$Name) {
  if (-not (Test-Path $Path)) { return $true }
  & $script:PythonExe $CloudIo upload --bucket $Config.bucket --object "$ObjectPrefix/$Name" --file $Path --client-id $Config.managedIdentityClientId
  if ($LASTEXITCODE -ne 0) {
    Write-Warning "Upload failed: $Name"
    return $false
  }
  return $true
}

function Write-ShardStatus([int]$FinalCode) {
  $StatusPath = Join-Path $OutputDir 'status.json'
  @{
    runId=$Config.runId
    shard=$Shard.id
    controllerEpoch=$Config.controllerEpoch
    repository=$Config.repository
    commit=$Config.commit
    releaseVersion=$Config.releaseVersion
    snapshotVersion=$Config.snapshotVersion
    contractDigest=$Shard.contractDigest
    variants=@($Shard.build.variants | ForEach-Object { $_.name })
    exitCode=$FinalCode
    completedAt=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
  } |
    ConvertTo-Json | Set-Content $StatusPath
  return $StatusPath
}

function Get-AzureStorageAccessToken {
  $ClientId = [Uri]::EscapeDataString([string]$Config.managedIdentityClientId)
  $TokenUri = "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https%3A%2F%2Fstorage.azure.com%2F&client_id=$ClientId"
  $Response = Invoke-RestMethod -Method Get -Uri $TokenUri -Headers @{Metadata='true'}
  if (-not $Response.access_token) { throw 'Azure managed identity returned no storage access token' }
  return [string]$Response.access_token
}

function Upload-AzureBlobPowerShell([string]$Path, [string]$ObjectName, [string]$AccessToken, [string]$ContentType) {
  $BucketParts = @(([string]$Config.bucket) -split '/', 2)
  if ($BucketParts.Count -ne 2) { throw "Invalid Azure bucket '$($Config.bucket)'" }
  $EncodedObject = (($ObjectName -split '/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
  $BlobUri = "https://$($BucketParts[0]).blob.core.windows.net/$($BucketParts[1])/$EncodedObject"
  $Headers = @{
    Authorization="Bearer $AccessToken"
    'x-ms-blob-type'='BlockBlob'
    'x-ms-date'=[DateTime]::UtcNow.ToString('R', [Globalization.CultureInfo]::InvariantCulture)
    'x-ms-version'='2021-12-02'
  }
  Invoke-WebRequest -Method Put -Uri $BlobUri -Headers $Headers -InFile $Path -ContentType $ContentType -UseBasicParsing | Out-Null
}

function Publish-BootstrapFailureWithoutPython([string]$Message, [string]$Details) {
  $Uploads = @()
  foreach ($Candidate in $Shards) {
    try {
      Set-ShardContext $Candidate
      $script:CurrentActive = $true
      $script:CurrentFinalized = $false
      Write-Phase 'worker-bootstrap' 'failed' $Message
      Write-BuildContent $Details
      $StatusPath = Write-ShardStatus 1
      $Uploads += [pscustomobject]@{Path=$BuildLog; Object="$ObjectPrefix/build.log"; ContentType='text/plain'}
      $Uploads += [pscustomobject]@{Path=$StatusPath; Object="$ObjectPrefix/status.json"; ContentType='application/json'}
    }
    catch {
      Write-Warning "Unable to prepare bootstrap failure for shard $($Candidate.id): $($_.Exception.Message)"
    }
    finally {
      $script:CurrentFinalized = $true
      $script:CurrentActive = $false
    }
  }
  if ($Uploads.Count -eq 0) { return $false }

  $PendingUploads = @($Uploads)
  $Deadline = [DateTime]::UtcNow.AddMinutes(15)
  $AttemptNumber = 0
  while ($PendingUploads.Count -gt 0 -and [DateTime]::UtcNow -lt $Deadline) {
    $AttemptNumber += 1
    $Remaining = @()
    try {
      $AccessToken = Get-AzureStorageAccessToken
      foreach ($Upload in $PendingUploads) {
        try {
          Upload-AzureBlobPowerShell $Upload.Path $Upload.Object $AccessToken $Upload.ContentType
        }
        catch {
          Write-Warning "Bootstrap blob upload failed for $($Upload.Object): $($_.Exception.Message)"
          $Remaining += $Upload
        }
      }
    }
    catch {
      Write-Warning "Azure bootstrap upload authentication failed: $($_.Exception.Message)"
      $Remaining = @($PendingUploads)
    }
    $PendingUploads = @($Remaining)
    if ($PendingUploads.Count -gt 0 -and [DateTime]::UtcNow -lt $Deadline) {
      Write-Output "[dl4j-phase] timestamp=$([DateTimeOffset]::UtcNow.ToString('o')) phase=bootstrap-log-upload status=waiting attempt=$AttemptNumber pending=$($PendingUploads.Count)"
      Start-Sleep -Seconds 15
    }
  }
  if ($PendingUploads.Count -gt 0) {
    Write-Warning "Unable to upload $($PendingUploads.Count) bootstrap failure blobs before the deadline"
    return $false
  }
  return $true
}

function Complete-Shard([int]$RequestedExitCode) {
  $FinalCode = $RequestedExitCode
  try {
    try {
      Stop-ShardLogging
    }
    catch {
      Write-Warning "Unable to stop shard logging for $($Shard.id): $($_.Exception.Message)"
      $FinalCode = 1
    }

    $Artifacts = @(
      [pscustomobject]@{Path=$BuildLog; Name='build.log'},
      [pscustomobject]@{Path=(Join-Path $OutputDir 'maven-repository.tar.gz'); Name='maven-repository.tar.gz'},
      [pscustomobject]@{Path=(Join-Path $OutputDir 'sdk-assets.tar.gz'); Name='sdk-assets.tar.gz'},
      [pscustomobject]@{Path=(Join-Path $OutputDir 'shard-manifest.json'); Name='shard-manifest.json'}
    )
    foreach ($Artifact in $Artifacts) {
      try {
        if (-not (Upload-IfPresent $Artifact.Path $Artifact.Name)) { $FinalCode = 1 }
      }
      catch {
        Write-Warning "Artifact finalization failed for $($Artifact.Name): $($_.Exception.Message)"
        $FinalCode = 1
      }
    }

    try {
      $StatusPath = Write-ShardStatus $FinalCode
      if (-not (Upload-IfPresent $StatusPath 'status.json')) {
        Write-Warning 'Final status upload was not acknowledged; the controller will reconcile the canonical blob'
        $FinalCode = 1
      }
    }
    catch {
      Write-Warning "Final status creation or upload failed for $($Shard.id): $($_.Exception.Message)"
      $FinalCode = 1
    }
  }
  catch {
    Write-Warning "Unexpected shard finalization failure for $($Shard.id): $($_.Exception.Message)"
    $FinalCode = 1
  }
  finally {
    $script:CurrentFinalized = $true
    $script:CurrentActive = $false
  }
  return [int]$FinalCode
}

function Start-KillWatchdog {
  $PythonExe = $script:PythonExe
  $ParentPid = $PID
  $script:KillWatchJob = Start-Job -Name "dl4j-release-kill-watchdog" -ArgumentList $PythonExe,$CloudIo,$Config.killSwitchBucket,$Config.killSwitchObject,$Config.controllerEpoch,$Config.managedIdentityClientId,$KillRequestedFile,$WatchdogStopFile,$WatchdogCloudPidFile,$BuildPidFile,$ParentPid -ScriptBlock {
    param($PythonExe,$CloudIo,$Bucket,$KillSwitchObject,$ControllerEpoch,$ClientId,$KillRequestedFile,$StopFile,$CloudPidFile,$BuildPidFile,$ParentPid)
    while (-not (Test-Path -LiteralPath $StopFile)) {
      $ProbeArguments = @($CloudIo, 'kill-enabled', '--bucket', $Bucket, '--object', $KillSwitchObject, '--controller-epoch', $ControllerEpoch, '--client-id', $ClientId)
      $Probe = Start-Process $PythonExe -ArgumentList $ProbeArguments -PassThru -NoNewWindow -RedirectStandardOutput "$CloudPidFile.out" -RedirectStandardError "$CloudPidFile.err"
      Set-Content -LiteralPath $CloudPidFile -Value $Probe.Id
      while (-not $Probe.HasExited) {
        if (Test-Path -LiteralPath $StopFile) {
          taskkill /PID $Probe.Id /T /F *> $null
          Remove-Item -LiteralPath $CloudPidFile -Force -ErrorAction SilentlyContinue
          return
        }
        Start-Sleep -Seconds 1
        $Probe.Refresh()
      }
      $State = $Probe.ExitCode
      Remove-Item -LiteralPath $CloudPidFile -Force -ErrorAction SilentlyContinue
      if ($State -ne 1) {
        $Reason = if ($State -eq 0) { 'enabled' } else { "unreadable-exit-$State" }
        Set-Content -LiteralPath $KillRequestedFile -Value $Reason
        if (Test-Path -LiteralPath $BuildPidFile) {
          $ActiveBuildPid = (Get-Content -Raw $BuildPidFile).Trim()
          if ($ActiveBuildPid -match '^\d+$') { taskkill /PID $ActiveBuildPid /T /F *> $null }
        }
        for ($Index = 0; $Index -lt 180; $Index++) {
          if (-not (Get-Process -Id $ParentPid -ErrorAction SilentlyContinue)) { return }
          Start-Sleep -Seconds 1
        }
        shutdown.exe /s /t 0 /f
        return
      }
      for ($Index = 0; $Index -lt 15; $Index++) {
        if (Test-Path -LiteralPath $StopFile) { return }
        Start-Sleep -Seconds 1
      }
    }
  }
}

function Install-CommonToolchains {
  Write-Phase 'toolchain-packages' 'started'
  Invoke-NativeChecked -Description 'Chocolatey toolchain installation' -SuccessCodes @(0, 1641, 3010) -Command {
    choco install -y --no-progress cmake git maven ninja temurin11 7zip msys2 rustup.install visualstudio2022buildtools visualstudio2022-workload-vctools
  }
  Write-Phase 'toolchain-packages' 'complete'
  $env:PATH = "C:\Program Files\Git\cmd;C:\Program Files\Git\bin;C:\tools\msys64\mingw64\bin;C:\tools\msys64\usr\bin;$env:PATH"
  $JavaHome = Get-ChildItem 'C:\Program Files\Eclipse Adoptium' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
  if (-not $JavaHome) {
    $JavaHome = Get-ChildItem 'C:\Program Files\Java' -Directory | Sort-Object Name -Descending | Select-Object -First 1
  }
  $env:JAVA_HOME = $JavaHome.FullName
  $env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
  $MavenHome = Get-ChildItem 'C:\ProgramData\chocolatey\lib\maven' -Directory -ErrorAction SilentlyContinue |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'bin\mvn.cmd') } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
  if (-not $MavenHome) {
    throw 'Chocolatey completed without installing an Apache Maven distribution'
  }
  $env:MAVEN_HOME = $MavenHome.FullName
  $env:M2_HOME = $MavenHome.FullName
  $env:PATH = "$($MavenHome.FullName)\bin;$env:PATH"
  $MavenExe = Join-Path $MavenHome.FullName 'bin\mvn.cmd'
  Invoke-NativeChecked -Description 'Maven toolchain validation' -Command {
    & $MavenExe --version
  }
  Write-Phase 'msys-toolchain' 'started'
  Invoke-NativeChecked -Description 'MSYS2 toolchain installation' -Command {
    & C:\tools\msys64\usr\bin\bash.exe -lc "pacman -S --needed --noconfirm base-devel git tar pkg-config unzip p7zip zip autoconf autoconf-archive automake patch make diffutils grep gzip mingw-w64-x86_64-make mingw-w64-x86_64-gnupg mingw-w64-x86_64-cmake mingw-w64-x86_64-nasm mingw-w64-x86_64-toolchain mingw-w64-x86_64-libtool mingw-w64-x86_64-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-libwinpthread-git mingw-w64-x86_64-SDL2 mingw-w64-x86_64-ragel mingw-w64-x86_64-sed mingw-w64-x86_64-ninja"
  }
  Write-Phase 'msys-toolchain' 'complete'
  $RustBinCandidates = @((Join-Path $env:CARGO_HOME 'bin'))
  if ($env:USERPROFILE) {
    $RustBinCandidates += (Join-Path $env:USERPROFILE '.cargo\bin')
  }
  $RustBinCandidates += (Join-Path $env:SystemRoot 'System32\config\systemprofile\.cargo\bin')
  $RustBin = $RustBinCandidates |
    Where-Object {
      (Test-Path -LiteralPath (Join-Path $_ 'rustup.exe')) -and
      (Test-Path -LiteralPath (Join-Path $_ 'cargo.exe'))
    } |
    Select-Object -First 1
  if (-not $RustBin) {
    throw "rustup.install completed without creating rustup.exe and cargo.exe in: $($RustBinCandidates -join ', ')"
  }
  $env:CARGO_HOME = Split-Path -Parent $RustBin
  $env:PATH = "$RustBin;$env:PATH"
  $RustupExe = Join-Path $RustBin 'rustup.exe'
  $CargoExe = Join-Path $RustBin 'cargo.exe'
  $CbindgenExe = Join-Path $RustBin 'cbindgen.exe'
  if (-not (Test-Path -LiteralPath $CbindgenExe)) {
    Invoke-NativeChecked -Description 'Rust GNU toolchain installation' -Command {
      & $RustupExe toolchain install stable-x86_64-pc-windows-gnu
    }
    Invoke-NativeChecked -Description 'Rust GNU toolchain selection' -Command {
      & $RustupExe default stable-x86_64-pc-windows-gnu
    }
    $env:CARGO_BUILD_TARGET = 'x86_64-pc-windows-gnu'
    Invoke-NativeChecked -Description 'cbindgen installation' -Command {
      & $CargoExe install --locked cbindgen
    }
  }
  $SccacheVersion = 'v0.15.0'
  $SccacheFile = "sccache-$SccacheVersion-x86_64-pc-windows-msvc"
  $SccacheSha256 = 'b0b257a164bf438b2dea134ca7ded41c100f59a64b3bf275a202f1e8102ab217'
  $SccacheDir = Join-Path $ToolchainRoot 'sccache'
  $SccacheExe = Join-Path $SccacheDir 'sccache.exe'
  if (-not (Test-Path -LiteralPath $script:WindowsTarExe)) {
    throw "Windows tar.exe was not found at $script:WindowsTarExe"
  }
  if (-not (Test-Path $SccacheExe)) {
    New-Item -ItemType Directory -Force -Path $SccacheDir | Out-Null
    $SccacheArchive = Join-Path $env:TEMP 'sccache.tar.gz'
    Invoke-WebRequest "https://github.com/mozilla/sccache/releases/download/$SccacheVersion/$SccacheFile.tar.gz" -OutFile $SccacheArchive -UseBasicParsing
    $ActualSccacheSha256 = (Get-FileHash -LiteralPath $SccacheArchive -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualSccacheSha256 -ne $SccacheSha256) {
      Remove-Item -LiteralPath $SccacheArchive -Force
      throw "sccache archive SHA-256 mismatch: expected $SccacheSha256, got $ActualSccacheSha256"
    }
    try {
      Invoke-NativeChecked -Description 'sccache archive extraction' -Command {
        & $script:WindowsTarExe -xzf $SccacheArchive -C $env:TEMP
      }
    }
    catch {
      Remove-Item -LiteralPath $SccacheArchive -Force
      throw
    }
    Copy-Item (Join-Path $env:TEMP "$SccacheFile\sccache.exe") $SccacheExe -Force
    Remove-Item -LiteralPath $SccacheArchive -Force
  }
  $env:PATH = "$SccacheDir;$env:PATH"
  $env:SCCACHE_DIR = Join-Path $SourceRoot 'sccache'
  $env:SCCACHE_CACHE_SIZE = '100G'
  $env:SCCACHE_IDLE_TIMEOUT = '0'
}

function Install-ShardCuda {
  if ($Shard.build.backend -ne 'cuda') { return }
  Write-Phase 'cuda-toolchain' 'started' "version=$($Shard.build.cudaVersion)"
  $env:CUDA_VERSION = $Shard.build.cudaVersion
  $CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$($Shard.build.cudaVersion)"
  if (-not (Test-Path "$CudaPath\bin\nvcc.exe")) {
    $Installer = Join-Path $ToolchainRoot 'install_cuda_windows.ps1'
    Invoke-WebRequest 'https://raw.githubusercontent.com/KonduitAI/cuda-install/master/.github/actions/install-cuda-windows/install_cuda_windows.ps1' -OutFile $Installer -UseBasicParsing
    & $Installer
  }
  $SparseVersion = if ($Shard.build.cudaVersion -eq '12.9') { '12.5.10.65' } else { '12.5.4.2' }
  if (-not (Test-Path "$CudaPath\include\cusparse_v2.h")) {
    $SparseZip = Join-Path $env:TEMP "cusparse-$($Shard.build.cudaVersion).zip"
    $SparseDir = Join-Path $ToolchainRoot "cusparse-$($Shard.build.cudaVersion)"
    Invoke-WebRequest "https://developer.download.nvidia.com/compute/cuda/redist/libcusparse/windows-x86_64/libcusparse-windows-x86_64-$SparseVersion-archive.zip" -OutFile $SparseZip -UseBasicParsing
    Expand-Archive $SparseZip $SparseDir -Force
    $SparseRoot = Get-ChildItem $SparseDir -Directory | Select-Object -First 1
    Copy-Item "$($SparseRoot.FullName)\include\*" "$CudaPath\include\" -Recurse -Force
    Copy-Item "$($SparseRoot.FullName)\lib\x64\*" "$CudaPath\lib\x64\" -Recurse -Force
    if (Test-Path "$($SparseRoot.FullName)\bin") {
      Copy-Item "$($SparseRoot.FullName)\bin\*" "$CudaPath\bin\" -Recurse -Force
    }
  }
  if (-not (Test-Path "$CudaPath\include\cusparse_v2.h")) {
    throw 'cuSPARSE installation is incomplete'
  }
  $env:CUDA_PATH = $CudaPath
  $env:CUDNN_ROOT_DIR = $CudaPath
  $env:PATH = "$CudaPath\bin;$CudaPath\libnvvp;$env:PATH"
  Write-Phase 'cuda-toolchain' 'complete' "version=$($Shard.build.cudaVersion)"
}

function Invoke-ShardBuild {
  Assert-NotKilled
  if (Test-KillSwitch) { throw 'Global kill switch is enabled or unreadable' }
  Install-ShardCuda
  Write-Phase 'source-checkout' 'started' "shard=$($Shard.id) commit=$($Config.commit)"
  Invoke-NativeChecked -Description 'Source clone' -Command {
    git -c core.autocrlf=false clone --filter=blob:none $Config.repository $SourceDir
  }
  Invoke-NativeChecked -Description 'Source line-ending configuration' -Command {
    git -C $SourceDir config core.autocrlf false
  }
  Invoke-NativeChecked -Description 'Pinned commit fetch' -Command {
    git -C $SourceDir fetch --depth=1 origin $Config.commit
  }
  Invoke-NativeChecked -Description 'Pinned commit checkout' -Command {
    git -C $SourceDir checkout --detach $Config.commit
  }
  $Actual = Invoke-NativeChecked -Description 'Pinned commit resolution' -Command {
    git -C $SourceDir rev-parse HEAD
  }
  if ($Actual.Trim() -ne $Config.commit) { throw "Commit mismatch: $Actual" }
  Write-Phase 'source-checkout' 'complete' "shard=$($Shard.id) commit=$($Config.commit)"

  $MavenOutput = Join-Path $OutputDir 'maven-repository'
  $SdkOutput = Join-Path $OutputDir 'sdk-assets'
  New-Item -ItemType Directory -Force -Path $MavenOutput,$SdkOutput | Out-Null
  $Arguments = @($BuildDriver, '--config', $ShardConfigFile, '--source', $SourceDir, '--repository', $ShardMavenRepo, '--maven-output', $MavenOutput, '--sdk-output', $SdkOutput)
  Write-Phase 'matrix-build' 'started' "shard=$($Shard.id)"
  $Process = Start-Process $script:PythonExe -ArgumentList $Arguments -RedirectStandardOutput $MatrixLog -RedirectStandardError $MatrixError -PassThru -NoNewWindow
  Set-Content -LiteralPath $BuildPidFile -Value $Process.Id
  try {
    while (-not $Process.HasExited) {
      Start-Sleep -Seconds 15
      Copy-NewLogContent $MatrixLog
      Copy-NewLogContent $MatrixError
      if ((Test-Path -LiteralPath $KillRequestedFile) -or (Test-KillSwitch)) {
        taskkill /PID $Process.Id /T /F *> $null
        throw 'Global kill switch stopped the build'
      }
      $Process.Refresh()
    }
  }
  finally {
    Remove-Item -LiteralPath $BuildPidFile -Force -ErrorAction SilentlyContinue
  }
  $Process.WaitForExit()
  $Process.Refresh()
  Copy-NewLogContent $MatrixLog
  Copy-NewLogContent $MatrixError
  $BuildExitCode = $Process.ExitCode
  if ($BuildExitCode -ne 0) { throw "Build failed with exit code $BuildExitCode" }
  Write-Phase 'matrix-build' 'complete' "shard=$($Shard.id)"

  Write-Phase 'artifact-packaging' 'started' "shard=$($Shard.id)"
  Invoke-NativeChecked -Description 'Maven repository packaging' -Command {
    & $script:WindowsTarExe -C $MavenOutput -czf (Join-Path $OutputDir 'maven-repository.tar.gz') .
  }
  Invoke-NativeChecked -Description 'SDK asset packaging' -Command {
    & $script:WindowsTarExe -C $SdkOutput -czf (Join-Path $OutputDir 'sdk-assets.tar.gz') .
  }
  $ManifestScript = 'import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); c=json.load(open(sys.argv[2])); s=c["shard"]; files=[]; [(lambda p: files.append({"path":p.relative_to(root).as_posix(),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"size":p.stat().st_size}))(p) for p in sorted(root.rglob("*")) if p.is_file()]; json.dump({"schemaVersion":1,"provider":"azure","runId":c["runId"],"shard":s["id"],"commit":c["commit"],"releaseVersion":c["releaseVersion"],"workloads":s["workloads"],"os":s["os"],"platform":s["build"]["javacppPlatform"],"backend":s["build"]["backend"],"variants":[v["name"] for v in s["build"]["variants"]],"files":files},open(root/"shard-manifest.json","w"),indent=2,sort_keys=True)'
  & $script:PythonExe -c $ManifestScript $OutputDir $ShardConfigFile
  if ($LASTEXITCODE -ne 0) { throw 'Shard manifest creation failed' }
  Write-Phase 'artifact-packaging' 'complete' "shard=$($Shard.id)"
}

function Invoke-CleanupStep([string]$Description, [scriptblock]$Action) {
  try {
    & $Action
  }
  catch {
    $script:ExitCode = 1
    Write-Warning "Cleanup step '$Description' failed: $($_.Exception.Message)"
  }
}

try {
  if (Test-Path -LiteralPath $AttemptFile) {
    $PriorAttempt = 0
    [void][int]::TryParse((Get-Content -Raw $AttemptFile).Trim(), [ref]$PriorAttempt)
    $Attempt = $PriorAttempt + 1
  }
  if ($Attempt -gt 1) {
    Remove-Item -LiteralPath $SourceRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
  New-Item -ItemType Directory -Force -Path $WorkRoot,$BootstrapRoot,$SourceRoot,$OutputRoot,$MavenRepoRoot,$ToolchainRoot,$env:CARGO_HOME,$env:RUSTUP_HOME,$env:TEMP | Out-Null
  Remove-Item -LiteralPath $KillRequestedFile,$WatchdogStopFile,$WatchdogCloudPidFile,$BuildPidFile -Force -ErrorAction SilentlyContinue
  Set-Content -LiteralPath $AttemptFile -Value $Attempt
  Set-Content -LiteralPath $WorkerStartedMarker -Value "started=$([DateTimeOffset]::UtcNow.ToString('o')) pid=$PID attempt=$Attempt"
  [IO.File]::WriteAllBytes($ConfigFile, [Convert]::FromBase64String($ConfigB64))
  [IO.File]::WriteAllBytes($BuildDriver, [Convert]::FromBase64String($BuildDriverB64))
  [IO.File]::WriteAllBytes($CloudIo, [Convert]::FromBase64String($CloudIoB64))
  $Config = Get-Content -Raw $ConfigFile | ConvertFrom-Json
  $Shards = if ($Config.shards) { @($Config.shards) } else { @($Config.shard) }
  if ($Shards.Count -eq 0) { throw 'Azure lane worker received no shards' }

  Set-ExecutionPolicy Bypass -Scope Process -Force
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Invoke-Expression ((New-Object Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
  }
  Invoke-NativeChecked -Description 'Python 3.12 installation' -SuccessCodes @(0, 1641, 3010) -Command {
    choco install -y --no-progress python312
  }
  $PythonInstall = Join-Path $env:SystemDrive 'Python312'
  $env:PATH = "${PythonInstall};${PythonInstall}\Scripts;C:\ProgramData\chocolatey\bin;$env:PATH"
  $script:PythonExe = Join-Path $PythonInstall 'python.exe'
  if (-not (Test-Path -LiteralPath $script:PythonExe)) {
    throw "Python 3.12 executable was not found at $script:PythonExe"
  }
  Write-Output "[dl4j-phase] timestamp=$([DateTimeOffset]::UtcNow.ToString('o')) phase=python-runtime status=ready executable=$script:PythonExe"
  Wait-ForCloudAccess

  $Pending = @()
  foreach ($Candidate in $Shards) {
    if (Test-RemoteShardSuccess $Candidate) {
      Write-Output "[dl4j-phase] timestamp=$([DateTimeOffset]::UtcNow.ToString('o')) phase=shard status=skipped shard=$($Candidate.id) reason=remote-success-checkpoint"
    }
    else {
      $Pending += $Candidate
    }
  }
  if ($Pending.Count -eq 0) {
    $ExitCode = 0
  }
  else {
    Set-ShardContext $Pending[0]
    $script:CurrentActive = $true
    $script:CurrentFinalized = $false
    Start-LaneLogging
    Start-ShardLogging
    Write-Phase 'worker' 'started' "lane=$($Config.laneId) pid=$PID attempt=$Attempt"
    Start-KillWatchdog
    Install-CommonToolchains
    Assert-NotKilled

    foreach ($Candidate in $Pending) {
      if ($Shard.id -ne $Candidate.id) {
        Set-ShardContext $Candidate
        $script:CurrentActive = $true
        $script:CurrentFinalized = $false
        Start-ShardLogging
      }
      $ShardExitCode = 0
      try {
        Write-Phase 'shard' 'started' "lane=$($Config.laneId) shard=$($Shard.id)"
        Invoke-ShardBuild
      }
      catch {
        Write-Phase 'shard' 'failed' $_.Exception.Message
        Write-BuildContent ($_ | Out-String)
        $ShardExitCode = 1
      }
      $FinalCode = Complete-Shard $ShardExitCode
      if ($FinalCode -ne 0) {
        throw "Shard $($Shard.id) failed with exit code $FinalCode"
      }
      Write-Output "[dl4j-phase] timestamp=$([DateTimeOffset]::UtcNow.ToString('o')) phase=shard status=complete lane=$($Config.laneId) shard=$($Shard.id)"
    }
    $ExitCode = 0
  }
}
catch {
  $BootstrapMessage = $_.Exception.Message
  $BootstrapError = $_ | Out-String
  Write-Output $BootstrapError
  if ($script:BuildLog) {
    try {
      Write-Phase 'worker' 'failed' $BootstrapMessage
      Write-BuildContent $BootstrapError
    }
    catch {
      Write-Warning "Unable to append the worker failure to the build log: $($_.Exception.Message)"
    }
  }
  else {
    $FallbackPython = Join-Path $env:SystemDrive 'Python312\python.exe'
    if (Test-Path -LiteralPath $FallbackPython) {
      try {
        $script:PythonExe = $FallbackPython
        Wait-ForCloudAccess
        foreach ($Candidate in $Shards) {
          try {
            Set-ShardContext $Candidate
            $script:CurrentActive = $true
            $script:CurrentFinalized = $false
            Write-Phase 'worker-bootstrap' 'failed' $BootstrapMessage
            Write-BuildContent $BootstrapError
            [void](Complete-Shard 1)
          }
          catch {
            Write-Warning "Unable to publish the Python bootstrap failure for shard $($Candidate.id): $($_.Exception.Message)"
            try { Stop-ShardLogging } catch { Write-Warning "Unable to stop fallback shard logging: $($_.Exception.Message)" }
            $script:CurrentFinalized = $true
            $script:CurrentActive = $false
          }
        }
      }
      catch {
        Write-Warning "Unable to initialize Python bootstrap failure publishing: $($_.Exception.Message)"
      }
    }
    if ($null -ne $Config -and $Shards.Count -gt 0) {
      try {
        if (-not (Publish-BootstrapFailureWithoutPython $BootstrapMessage $BootstrapError)) {
          Write-Warning 'The direct Azure bootstrap failure upload did not complete'
        }
      }
      catch {
        Write-Warning "Unable to publish bootstrap failure directly to Azure Blob Storage: $($_.Exception.Message)"
      }
    }
    else {
      Write-Warning 'Bootstrap failed before the Azure shard configuration could be loaded'
    }
  }
  $ExitCode = 1
}
finally {
  Invoke-CleanupStep 'active shard finalization' {
    if ($script:CurrentActive -and -not $script:CurrentFinalized) {
      $FinalCode = Complete-Shard $script:ExitCode
      if ($FinalCode -ne 0) { $script:ExitCode = 1 }
    }
  }
  Invoke-CleanupStep 'kill watchdog stop signal' {
    if ($script:KillWatchJob) {
      New-Item -ItemType File -Force -Path $WatchdogStopFile | Out-Null
    }
  }
  Invoke-CleanupStep 'kill watchdog job cleanup' {
    if ($script:KillWatchJob) {
      Wait-Job -Job $script:KillWatchJob -Timeout 35 -ErrorAction SilentlyContinue | Out-Null
      if ($script:KillWatchJob.State -ne 'Completed') {
        if (Test-Path -LiteralPath $WatchdogCloudPidFile) {
          $CloudPid = (Get-Content -Raw $WatchdogCloudPidFile).Trim()
          if ($CloudPid -match '^\d+$') { taskkill /PID $CloudPid /T /F *> $null }
        }
        Stop-Job -Job $script:KillWatchJob -ErrorAction SilentlyContinue
      }
      Remove-Job -Job $script:KillWatchJob -Force -ErrorAction SilentlyContinue
      $script:KillWatchJob = $null
    }
  }
  Invoke-CleanupStep 'PowerShell transcript cleanup' {
    if ($script:TranscriptStarted) {
      Stop-Transcript | Out-Null
      $script:TranscriptStarted = $false
    }
  }
  Invoke-CleanupStep 'lane log cleanup' {
    Stop-LaneLogging
  }
  Invoke-CleanupStep 'scheduled worker cleanup' {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
  }
  Invoke-CleanupStep 'VM shutdown' {
    try {
      $global:LASTEXITCODE = $null
      & shutdown.exe /s /t 0 /f
      if ($null -ne $global:LASTEXITCODE -and $global:LASTEXITCODE -ne 0) {
        throw "shutdown.exe failed with exit code $global:LASTEXITCODE"
      }
    }
    catch {
      Write-Warning "shutdown.exe was unsuccessful; using Stop-Computer: $($_.Exception.Message)"
      Stop-Computer -Force
    }
  }
}
exit $ExitCode
