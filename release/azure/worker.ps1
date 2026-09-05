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
    $Started = $false
    $LastResult = $null
    $LastState = $null
    for ($StartAttempt = 1; $StartAttempt -le 3 -and -not $Started; $StartAttempt++) {
      Write-Output "Starting scheduled worker attempt $StartAttempt of 3"
      Start-ScheduledTask -TaskName $TaskName
      $Deadline = [DateTime]::UtcNow.AddSeconds(90)
      while (-not (Test-Path -LiteralPath $WorkerStartedMarker) -and [DateTime]::UtcNow -lt $Deadline) {
        Start-Sleep -Seconds 2
      }
      if (Test-Path -LiteralPath $WorkerStartedMarker) {
        $Started = $true
        break
      }
      $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
      $Info = Get-ScheduledTaskInfo -TaskName $TaskName -ErrorAction SilentlyContinue
      $LastState = if ($Task) { $Task.State } else { 'missing' }
      $LastResult = if ($Info) { $Info.LastTaskResult } else { 'unavailable' }
      Write-Warning "Scheduled worker attempt $StartAttempt did not start (state=$LastState lastResult=$LastResult)"
      Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
      Start-Sleep -Seconds 5
    }
    if (-not $Started) {
      throw "Scheduled worker did not start after 3 attempts (state=$LastState lastResult=$LastResult)"
    }
  }
  catch {
    Write-Error "Windows worker registration failed: $($_ | Out-String)"
    exit 1
  }
  exit 0
}

$ConfigB64 = '__DL4J_WORKER_CONFIG_B64__'
$BuildDriverB64 = '__DL4J_BUILD_DRIVER_B64__'
$CloudIoB64 = '__DL4J_CLOUD_IO_B64__'
$DependencyCacheB64 = '__DL4J_DEPENDENCY_CACHE_B64__'
$NativePlatformScriptB64 = '__DL4J_NATIVE_PLATFORM_SCRIPT_B64__'
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
$DependencyCache = Join-Path $BootstrapRoot 'dependency-cache.py'
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
  $Invocation = [pscustomobject]@{ ExitCode = $null }
  try {
    $ErrorActionPreference = 'Continue'
    # A scriptblock invoked with '&' has child scope, so its native command's
    # LASTEXITCODE does not reliably flow back to this helper. Run the caller's
    # block dot-sourced inside an isolated scope and copy the exit code through
    # a mutable holder before that scope ends.
    & {
      . $Command
      $Invocation.ExitCode = $LASTEXITCODE
    }
    $Code = $Invocation.ExitCode
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

function Invoke-WebRequestWithRetry {
  param(
    [Parameter(Mandatory=$true)][string]$Uri,
    [Parameter(Mandatory=$true)][string]$OutFile,
    [Parameter(Mandatory=$true)][string]$Description,
    [int]$MaxAttempts = 6
  )

  for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
    try {
      if (Test-Path -LiteralPath $OutFile) {
        Remove-Item -LiteralPath $OutFile -Force
      }
      Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $OutFile
      return
    }
    catch {
      if (Test-Path -LiteralPath $OutFile) {
        Remove-Item -LiteralPath $OutFile -Force
      }
      if ($Attempt -ge $MaxAttempts) {
        throw "$Description download failed after $MaxAttempts attempts: $($_.Exception.Message)"
      }
      $DelaySeconds = [Math]::Min(60, 5 * [Math]::Pow(2, $Attempt - 1))
      Write-Phase 'tool-download' 'retrying' "name=$Description attempt=$Attempt maxAttempts=$MaxAttempts delaySeconds=$DelaySeconds"
      Start-Sleep -Seconds $DelaySeconds
    }
  }
}

function Get-ToolchainIdentity {
  param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][object]$Contract
  )
  $Payload = [ordered]@{
    schemaVersion = 1
    name = $Name
    contract = $Contract
  } | ConvertTo-Json -Compress -Depth 20
  $Sha256 = [Security.Cryptography.SHA256]::Create()
  try {
    $Bytes = [Text.Encoding]::UTF8.GetBytes($Payload)
    return (($Sha256.ComputeHash($Bytes) | ForEach-Object { $_.ToString('x2') }) -join '')
  }
  finally {
    $Sha256.Dispose()
  }
}

function Test-ToolchainCacheConfigured {
  return $null -ne $Config.compilerCache -and
    $null -ne $Config.compilerCache.toolchainCache -and
    -not [string]::IsNullOrWhiteSpace([string]$Config.compilerCache.account) -and
    -not [string]::IsNullOrWhiteSpace([string]$Config.compilerCache.container) -and
    -not [string]::IsNullOrWhiteSpace([string]$Config.compilerCache.toolchainCache.keyPrefix)
}

function Restore-ToolchainDependency {
  param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$Identity,
    [Parameter(Mandatory=$true)][string]$Destination
  )
  if (-not (Test-ToolchainCacheConfigured)) { return $false }
  $Bucket = "$($Config.compilerCache.account)/$($Config.compilerCache.container)"
  $Prefix = [string]$Config.compilerCache.toolchainCache.keyPrefix
  Write-Phase 'dependency-cache' 'started' "dependency=$Name identity=$Identity operation=restore"
  $PreviousErrorActionPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'Continue'
    $CacheOutput = & $script:PythonExe $DependencyCache restore --cloud-io $CloudIo --bucket $Bucket --prefix $Prefix --name $Name --identity $Identity --destination $Destination --client-id $Config.managedIdentityClientId 2>&1
    $Code = $LASTEXITCODE
  }
  finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
  }
  $CacheOutput | ForEach-Object { Write-Output $_ }
  if ($Code -eq 0) {
    Write-Phase 'dependency-cache' 'complete' "dependency=$Name identity=$Identity operation=restore result=hit"
    return $true
  }
  if ($Code -eq 3) {
    Write-Phase 'dependency-cache' 'complete' "dependency=$Name identity=$Identity operation=restore result=miss"
    return $false
  }
  Write-Phase 'dependency-cache' 'failed' "dependency=$Name identity=$Identity operation=restore exitCode=$Code"
  throw "Toolchain cache restore failed for $Name/$Identity with exit code $Code"
}

function Publish-ToolchainDependency {
  param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$Identity,
    [Parameter(Mandatory=$true)][string]$Source
  )
  if (-not (Test-ToolchainCacheConfigured)) { return }
  $Bucket = "$($Config.compilerCache.account)/$($Config.compilerCache.container)"
  $Prefix = [string]$Config.compilerCache.toolchainCache.keyPrefix
  Write-Phase 'dependency-cache' 'started' "dependency=$Name identity=$Identity operation=publish"
  $PreviousErrorActionPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = 'Continue'
    $CacheOutput = & $script:PythonExe $DependencyCache publish --cloud-io $CloudIo --bucket $Bucket --prefix $Prefix --name $Name --identity $Identity --source $Source --client-id $Config.managedIdentityClientId 2>&1
    $Code = $LASTEXITCODE
  }
  finally {
    $ErrorActionPreference = $PreviousErrorActionPreference
  }
  $CacheOutput | ForEach-Object { Write-Output $_ }
  if ($Code -ne 0) {
    Write-Phase 'dependency-cache' 'failed' "dependency=$Name identity=$Identity operation=publish exitCode=$Code"
    throw "Toolchain cache publish failed for $Name/$Identity with exit code $Code"
  }
  Write-Phase 'dependency-cache' 'complete' "dependency=$Name identity=$Identity operation=publish result=stored"
}

function Invoke-KillSwitchProbe {
  & $script:PythonExe $CloudIo kill-enabled --bucket $Config.killSwitchBucket --object $Config.runKillSwitchObject --emergency-object $Config.killSwitchObject --controller-epoch $Config.controllerEpoch --client-id $Config.managedIdentityClientId *> $null
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

function Publish-MavenRepository(
  [string]$Path,
  [string]$SdkAssets,
  [string]$ConfigPath,
  [bool]$AllowMissingUnclassified
) {
  $Prefix = ([string]$Config.mavenRepositoryPrefix).TrimEnd('/')
  if ([string]::IsNullOrWhiteSpace($Prefix)) {
    throw 'Azure worker received no stable Maven repository prefix'
  }
  $AccountingPath = Join-Path $OutputDir 'maven-publish.json'
  $Publisher = Join-Path $SourceDir 'release\azure\maven-publish.py'
  $CentralRepository = Join-Path $SourceDir 'release\central\repository.py'
  Invoke-NativeChecked -Description 'Direct stable Maven publication' -Command {
    $PublisherArguments = @(
      $Publisher,
      '--repository', $Path,
      '--sdk-assets', $SdkAssets,
      '--config', $ConfigPath,
      '--central-repository', $CentralRepository,
      '--cloud-io', $CloudIo,
      '--bucket', $Config.bucket,
      '--repository-prefix', $Prefix,
      '--client-id', $Config.managedIdentityClientId,
      '--run-id', $Config.runId,
      '--shard', $Shard.id,
      '--release-version', $Config.releaseVersion,
      '--commit', $Config.commit,
      '--accounting', $AccountingPath
    )
    if ($AllowMissingUnclassified) {
      $PublisherArguments += '--allow-missing-unclassified'
    }
    & $script:PythonExe @PublisherArguments
  }
  return $AccountingPath
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
      [pscustomobject]@{Path=(Join-Path $OutputDir 'build-benchmark.json'); Name='build-benchmark.json'},
      [pscustomobject]@{Path=(Join-Path $OutputDir 'maven-publish.json'); Name='maven-publish.json'},
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
  $script:KillWatchJob = Start-Job -Name "dl4j-release-kill-watchdog" -ArgumentList $PythonExe,$CloudIo,$Config.killSwitchBucket,$Config.runKillSwitchObject,$Config.killSwitchObject,$Config.controllerEpoch,$Config.managedIdentityClientId,$KillRequestedFile,$WatchdogStopFile,$WatchdogCloudPidFile,$BuildPidFile,$ParentPid -ScriptBlock {
    param($PythonExe,$CloudIo,$Bucket,$KillSwitchObject,$EmergencyKillSwitchObject,$ControllerEpoch,$ClientId,$KillRequestedFile,$StopFile,$CloudPidFile,$BuildPidFile,$ParentPid)
    while (-not (Test-Path -LiteralPath $StopFile)) {
      $ProbeArguments = @($CloudIo, 'kill-enabled', '--bucket', $Bucket, '--object', $KillSwitchObject, '--emergency-object', $EmergencyKillSwitchObject, '--controller-epoch', $ControllerEpoch, '--client-id', $ClientId)
      $Probe = Start-Process $PythonExe -ArgumentList $ProbeArguments -PassThru -NoNewWindow -RedirectStandardOutput "$CloudPidFile.out" -RedirectStandardError "$CloudPidFile.err"
      $null = $Probe.Handle
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
      $Probe.WaitForExit()
      $State = $Probe.ExitCode
      if ($null -eq $State) { $State = -1 }
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

function Import-VisualStudioEnvironment {
  Write-Phase 'visual-studio-environment' 'started'
  $VsWhereRoot = ${env:ProgramFiles(x86)}
  if (-not $VsWhereRoot) { $VsWhereRoot = $env:ProgramFiles }
  $VsWhere = Join-Path $VsWhereRoot 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path -LiteralPath $VsWhere)) {
    throw "Visual Studio locator was not found at $VsWhere"
  }
  $VsInstall = (& $VsWhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
  if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($VsInstall)) {
    throw 'Visual Studio 2022 C++ Build Tools installation was not found'
  }
  $VsInstall = $VsInstall.Trim()
  $VcVarsVersion = $null
  if ($Shard.build.backend -eq 'cuda' -and $Shard.build.cudaVersion -in @('12.6', '13.1')) {
    $SupportedToolsetVersion = '14.38'
    $SupportedToolsetComponent = 'Microsoft.VisualStudio.Component.VC.14.38.17.8.x86.x64'
    $ToolsetRoot = Join-Path $VsInstall 'VC\Tools\MSVC'
    $SupportedToolset = Get-ChildItem -LiteralPath $ToolsetRoot -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -like "$SupportedToolsetVersion.*" } |
      Sort-Object Name -Descending |
      Select-Object -First 1
    if ($null -eq $SupportedToolset) {
      $VsSetup = Join-Path $VsWhereRoot 'Microsoft Visual Studio\Installer\setup.exe'
      if (-not (Test-Path -LiteralPath $VsSetup)) {
        throw "Visual Studio installer was not found at $VsSetup"
      }
      $SetupArguments = @(
        'modify', '--installPath', "`"$VsInstall`"",
        '--channelId', 'VisualStudio.17.Release',
        '--productId', 'Microsoft.VisualStudio.Product.BuildTools',
        '--add', $SupportedToolsetComponent,
        '--quiet', '--norestart'
      )
      $Setup = Start-Process -FilePath $VsSetup -ArgumentList $SetupArguments -Wait -PassThru
      if ($Setup.ExitCode -notin @(0, 3010)) {
        throw "Installing $SupportedToolsetComponent failed with exit code $($Setup.ExitCode)"
      }
      $SupportedToolset = Get-ChildItem -LiteralPath $ToolsetRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "$SupportedToolsetVersion.*" } |
        Sort-Object Name -Descending |
        Select-Object -First 1
    }
    if ($null -eq $SupportedToolset) {
      throw "CUDA $($Shard.build.cudaVersion) requires MSVC 193x, but toolset $SupportedToolsetVersion was not installed"
    }
    $VcVarsVersion = $SupportedToolsetVersion
  }
  $VcVars = Join-Path $VsInstall 'VC\Auxiliary\Build\vcvars64.bat'
  if (-not (Test-Path -LiteralPath $VcVars)) {
    throw "Visual Studio x64 environment script was not found at $VcVars"
  }
  $VcVarsCommand = "`"$VcVars`""
  if ($VcVarsVersion) { $VcVarsCommand += " -vcvars_ver=$VcVarsVersion" }
  $EnvironmentLines = & $env:ComSpec /d /s /c "$VcVarsCommand >nul && set"
  if ($LASTEXITCODE -ne 0) {
    throw "Visual Studio x64 environment initialization failed with exit code $LASTEXITCODE"
  }
  foreach ($Line in $EnvironmentLines) {
    $Separator = $Line.IndexOf('=')
    if ($Separator -le 0) { continue }
    $Name = $Line.Substring(0, $Separator)
    $Value = $Line.Substring($Separator + 1)
    [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
  }
  if (-not $env:VCINSTALLDIR -or -not $env:INCLUDE -or -not $env:LIB) {
    throw 'Visual Studio environment is missing VCINSTALLDIR, INCLUDE, or LIB'
  }
  if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'Visual Studio environment did not expose cl.exe on PATH'
  }
  if ($VcVarsVersion -and $env:VCToolsVersion -notlike "$VcVarsVersion.*") {
    throw "Requested MSVC $VcVarsVersion for CUDA $($Shard.build.cudaVersion), but vcvars selected $($env:VCToolsVersion)"
  }
  Write-Phase 'visual-studio-environment' 'complete' "installation=$VsInstall toolset=$($env:VCToolsVersion)"
}

function Install-CommonToolchains {
  Write-Phase 'toolchain-packages' 'started'
  $MsysIdentity = Get-ToolchainIdentity 'msys2' ([ordered]@{
    cacheSchema = 2
    version = '20260611'
    platform = 'windows-x86_64'
    packages = 'base-devel,git,tar,pkg-config,unzip,p7zip,zip,autoconf,autoconf-archive,automake,patch,make,diffutils,grep,gzip,mingw-w64-x86_64-make,mingw-w64-x86_64-gnupg,mingw-w64-x86_64-cmake,mingw-w64-x86_64-nasm,mingw-w64-x86_64-toolchain,mingw-w64-x86_64-libtool,mingw-w64-x86_64-gcc,mingw-w64-x86_64-gcc-fortran,mingw-w64-x86_64-libwinpthread-git,mingw-w64-x86_64-SDL2,mingw-w64-x86_64-ragel,mingw-w64-x86_64-sed,mingw-w64-x86_64-ninja,mingw-w64-x86_64-vulkan-headers,mingw-w64-x86_64-vulkan-loader'
  })
  # Restore-ToolchainDependency emits phase lines as well as its Boolean result.
  # Consume the lines while retaining the actual result; assigning the raw
  # pipeline would turn any log line into a truthy array and create a false hit.
  $script:LastToolchainRestoreResult = $false
  Restore-ToolchainDependency 'msys2' $MsysIdentity 'C:\tools\msys64' | ForEach-Object {
    if ($_ -is [bool]) {
      $script:LastToolchainRestoreResult = [bool]$_
    } else {
      Write-Output $_
    }
  }
  $MsysRestored = $script:LastToolchainRestoreResult
  if ($MsysRestored) {
    # A cache marker is not enough: Rust's GNU target requires the actual
    # MinGW binutils and compiler, especially dlltool.exe, to be present.
    $MsysCacheUsable =
      (Test-Path -LiteralPath 'C:\tools\msys64\usr\bin\bash.exe') -and
      (Test-Path -LiteralPath 'C:\tools\msys64\usr\bin\pacman.exe') -and
      (Test-Path -LiteralPath 'C:\tools\msys64\mingw64\bin\dlltool.exe') -and
      (Test-Path -LiteralPath 'C:\tools\msys64\mingw64\bin\gcc.exe') -and
      (Test-Path -LiteralPath 'C:\tools\msys64\mingw64\bin\g++.exe')
    if (-not $MsysCacheUsable) {
      Write-Phase 'dependency-cache' 'complete' "dependency=msys2 identity=$MsysIdentity operation=restore result=invalid"
      Remove-Item -LiteralPath 'C:\tools\msys64' -Recurse -Force -ErrorAction SilentlyContinue
      $MsysRestored = $false
    }
  }
  $RustIdentity = Get-ToolchainIdentity 'rust-cbindgen' ([ordered]@{
    cacheSchema = 2
    rust = 'stable-x86_64-pc-windows-gnu'
    cbindgen = 'latest-locked'
    platform = 'windows-x86_64'
  })
  $script:LastToolchainRestoreResult = $false
  Restore-ToolchainDependency 'rust-cbindgen' $RustIdentity $ToolchainRoot | ForEach-Object {
    if ($_ -is [bool]) {
      $script:LastToolchainRestoreResult = [bool]$_
    } else {
      Write-Output $_
    }
  }
  $RustRestored = $script:LastToolchainRestoreResult
  if ($RustRestored) {
    # A cache index is not sufficient evidence that the archive is usable.
    # Validate the executable contract before suppressing the Rust installer;
    # this prevents an incomplete archive from becoming a persistent false hit.
    $RustCacheProbeCandidates = @(
      (Join-Path $env:CARGO_HOME 'bin'),
      (Join-Path $ToolchainRoot 'cargo\bin'),
      (Join-Path $ToolchainRoot 'bin'),
      (Join-Path $ToolchainRoot 'rust\bin')
    ) | Select-Object -Unique
    $RustCacheUsable = $false
    foreach ($RustCacheCandidate in $RustCacheProbeCandidates) {
      $HasCargo = Test-Path -LiteralPath (Join-Path $RustCacheCandidate 'cargo.exe')
      $HasRust = (Test-Path -LiteralPath (Join-Path $RustCacheCandidate 'rustup.exe')) -or
        (Test-Path -LiteralPath (Join-Path $RustCacheCandidate 'rustc.exe'))
      if ($HasCargo -and $HasRust) {
        $RustCacheUsable = $true
        break
      }
    }
    if (-not $RustCacheUsable) {
      Write-Phase 'dependency-cache' 'complete' "dependency=rust-cbindgen identity=$RustIdentity operation=restore result=invalid"
      Remove-Item -LiteralPath $ToolchainRoot -Recurse -Force -ErrorAction SilentlyContinue
      New-Item -ItemType Directory -Force -Path $ToolchainRoot,$env:CARGO_HOME,$env:RUSTUP_HOME | Out-Null
      $RustRestored = $false
    }
  }
  $ToolchainInstalled = $false
  for ($ChocolateyAttempt = 1; $ChocolateyAttempt -le 8 -and -not $ToolchainInstalled; $ChocolateyAttempt++) {
    try {
      $ChocolateyPackages = @('cmake','git','maven','ninja','temurin11','7zip','visualstudio2022buildtools','visualstudio2022-workload-vctools')
      if (-not $RustRestored) { $ChocolateyPackages += 'rust' }
      Invoke-NativeChecked -Description 'Chocolatey toolchain installation' -SuccessCodes @(0, 1641, 3010) -Command {
        choco install -y --no-progress @ChocolateyPackages
      }
      $ToolchainInstalled = $true
    }
    catch {
      if ($ChocolateyAttempt -ge 8) { throw }
      $ChocolateyBackoff = [Math]::Min(60, 15 * $ChocolateyAttempt)
      Write-Warning "Chocolatey toolchain installation attempt $ChocolateyAttempt failed: $($_.Exception.Message)"
      Write-Phase 'toolchain-packages' 'retrying' "group=common attempt=$ChocolateyAttempt backoffSeconds=$ChocolateyBackoff"
      Start-Sleep -Seconds $ChocolateyBackoff
    }
  }
  if (-not $MsysRestored) {
    $MsysInstalled = $false
    for ($MsysAttempt = 1; $MsysAttempt -le 8 -and -not $MsysInstalled; $MsysAttempt++) {
      try {
        Invoke-NativeChecked -Description 'Chocolatey MSYS2 installation' -SuccessCodes @(0, 1641, 3010) -Command {
          choco install -y --no-progress msys2 --params "/NoUpdate"
        }
        $MsysInstalled = $true
      }
      catch {
        if ($MsysAttempt -ge 8) { throw }
        $MsysBackoff = [Math]::Min(60, 15 * $MsysAttempt)
        Write-Warning "Chocolatey MSYS2 installation attempt $MsysAttempt failed: $($_.Exception.Message)"
        Write-Phase 'toolchain-packages' 'retrying' "group=msys2 attempt=$MsysAttempt backoffSeconds=$MsysBackoff"
        Start-Sleep -Seconds $MsysBackoff
      }
    }
  }
  Write-Phase 'toolchain-packages' 'complete'
  Import-VisualStudioEnvironment
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
  if (-not $MsysRestored) {
    Invoke-NativeChecked -Description 'MSYS2 toolchain installation' -Command {
      & C:\tools\msys64\usr\bin\bash.exe -lc "pacman-key --init && pacman-key --populate msys2 && pacman -S --needed --noconfirm base-devel git tar pkg-config unzip p7zip zip autoconf autoconf-archive automake patch make diffutils grep gzip mingw-w64-x86_64-make mingw-w64-x86_64-gnupg mingw-w64-x86_64-cmake mingw-w64-x86_64-nasm mingw-w64-x86_64-toolchain mingw-w64-x86_64-libtool mingw-w64-x86_64-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-libwinpthread-git mingw-w64-x86_64-SDL2 mingw-w64-x86_64-ragel mingw-w64-x86_64-sed mingw-w64-x86_64-ninja mingw-w64-x86_64-vulkan-headers mingw-w64-x86_64-vulkan-loader"
    }
    Publish-ToolchainDependency 'msys2' $MsysIdentity 'C:\tools\msys64'
  }
  Write-Phase 'msys-toolchain' 'complete'
  # Keep runtime discovery aligned with the cache validator. Cached Rust
  # toolchains may be rooted directly under toolchains\bin/rust\bin rather
  # than CARGO_HOME\bin; omitting those paths turns a valid cache into a
  # bootstrap failure after restore.
  $RustBinCandidates = @(
    (Join-Path $env:CARGO_HOME 'bin'),
    (Join-Path $ToolchainRoot 'cargo\bin'),
    (Join-Path $ToolchainRoot 'bin'),
    (Join-Path $ToolchainRoot 'rust\bin')
  ) | Select-Object -Unique
  if ($env:USERPROFILE) {
    $RustBinCandidates += (Join-Path $env:USERPROFILE '.cargo\bin')
  }
  $RustBinCandidates += (Join-Path $env:SystemRoot 'System32\config\systemprofile\.cargo\bin')
  $RustBinCandidates += 'C:\ProgramData\chocolatey\bin'
  $RustBinCandidates += @(Get-ChildItem 'C:\Program Files\Rust*' -Directory -ErrorAction SilentlyContinue |
    ForEach-Object { Join-Path $_.FullName 'bin' })
  $RustBin = $RustBinCandidates |
    Where-Object {
      (Test-Path -LiteralPath (Join-Path $_ 'cargo.exe')) -and
      ((Test-Path -LiteralPath (Join-Path $_ 'rustup.exe')) -or
       (Test-Path -LiteralPath (Join-Path $_ 'rustc.exe')))
    } |
    Select-Object -First 1
  if (-not $RustBin -and -not $RustRestored) {
    # Keep a deterministic fallback for images where the Chocolatey Rust
    # package is unavailable. The official installer honors the worker's
    # CARGO_HOME/RUSTUP_HOME and creates the GNU host toolchain in-place.
    $RustupInitExe = Join-Path $ToolchainRoot 'rustup-init.exe'
    Invoke-WebRequestWithRetry `
      -Uri 'https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe' `
      -OutFile $RustupInitExe `
      -Description 'Rustup host installer'
    Invoke-NativeChecked -Description 'Rustup host bootstrap' -Command {
      & $RustupInitExe -y --default-toolchain stable-x86_64-pc-windows-gnu --profile minimal --no-modify-path
    }
    Remove-Item -LiteralPath $RustupInitExe -Force -ErrorAction SilentlyContinue
    $RustBin = $RustBinCandidates |
      Where-Object {
        (Test-Path -LiteralPath (Join-Path $_ 'cargo.exe')) -and
        ((Test-Path -LiteralPath (Join-Path $_ 'rustup.exe')) -or
         (Test-Path -LiteralPath (Join-Path $_ 'rustc.exe')))
      } |
      Select-Object -First 1
  }
  if (-not $RustBin) {
    throw "Rust bootstrap completed without creating cargo.exe plus rustup.exe or rustc.exe in: $($RustBinCandidates -join ', ')"
  }
  $RustBinPath = [IO.Path]::GetFullPath([string]$RustBin).TrimEnd('\')
  $RustToolchainRoot = Join-Path $ToolchainRoot 'rust'
  if ($RustBinPath.StartsWith('C:\ProgramData\chocolatey', [StringComparison]::OrdinalIgnoreCase)) {
    # Chocolatey's Rust package exposes machine-wide shims and keeps the real
    # GNU toolchain outside the cache source. Stage that toolchain under the
    # worker root so cbindgen and the compiler are published together.
    $ChocolateyRustRoot = 'C:\ProgramData\chocolatey\lib\rust\tools'
    if (-not (Test-Path -LiteralPath (Join-Path $ChocolateyRustRoot 'bin\cargo.exe'))) {
      throw "Chocolatey Rust installation is missing cargo.exe under $ChocolateyRustRoot"
    }
    Remove-Item -LiteralPath $RustToolchainRoot -Recurse -Force -ErrorAction SilentlyContinue
    Copy-Item -LiteralPath $ChocolateyRustRoot -Destination $RustToolchainRoot -Recurse -Force
    $RustBin = Join-Path $RustToolchainRoot 'bin'
    $RustBinPath = $RustBin
    $env:CARGO_HOME = Join-Path $ToolchainRoot 'cargo'
    $env:RUSTUP_HOME = Join-Path $ToolchainRoot 'rustup'
  } else {
    $env:CARGO_HOME = Split-Path -Parent $RustBin
  }
  $RustInstallBin = Join-Path $env:CARGO_HOME 'bin'
  New-Item -ItemType Directory -Force -Path $RustInstallBin | Out-Null
  $env:PATH = "$RustInstallBin;$RustBin;$env:PATH"
  $RustupExe = Join-Path $RustBin 'rustup.exe'
  if (-not (Test-Path -LiteralPath $RustupExe)) {
    $RustupExe = $null
  }
  $RustcExe = Join-Path $RustBin 'rustc.exe'
  $CargoExe = Join-Path $RustBin 'cargo.exe'
  $CbindgenExe = Join-Path $RustInstallBin 'cbindgen.exe'
  if (-not (Test-Path -LiteralPath $CbindgenExe)) {
    $CbindgenExe = Join-Path $RustBin 'cbindgen.exe'
  }
  if (-not (Test-Path -LiteralPath $CbindgenExe)) {
    if ($RustupExe) {
      Invoke-NativeChecked -Description 'Rust GNU toolchain installation' -Command {
        & $RustupExe toolchain install stable-x86_64-pc-windows-gnu
      }
      Invoke-NativeChecked -Description 'Rust GNU toolchain selection' -Command {
        & $RustupExe default stable-x86_64-pc-windows-gnu
      }
    } elseif (-not (Test-Path -LiteralPath $RustcExe)) {
      throw "Rust installation at $RustBin has cargo.exe but neither rustup.exe nor rustc.exe"
    }
    $env:CARGO_BUILD_TARGET = 'x86_64-pc-windows-gnu'
    Invoke-NativeChecked -Description 'cbindgen installation' -Command {
      & $CargoExe install --locked cbindgen
    }
    $CbindgenExe = Join-Path $RustInstallBin 'cbindgen.exe'
    if (-not (Test-Path -LiteralPath $CbindgenExe)) {
      $CbindgenExe = Join-Path $RustBin 'cbindgen.exe'
    }
  }
  if (-not $RustRestored) {
    Publish-ToolchainDependency 'rust-cbindgen' $RustIdentity $ToolchainRoot
  }
  $SccacheVersion = 'v0.17.0'
  $SccacheFile = "sccache-$SccacheVersion-x86_64-pc-windows-msvc"
  $SccacheSha256 = 'caf1932d76a909c909b7a2e41443cdfe3c79a49a380da1a22fa422e1d00d3ca7'
  $SccacheDir = Join-Path $ToolchainRoot 'sccache'
  $SccacheExe = Join-Path $SccacheDir 'sccache.exe'
  $SccacheIdentity = Get-ToolchainIdentity 'sccache' ([ordered]@{
    version = $SccacheVersion
    archive = "$SccacheFile.tar.gz"
    sha256 = $SccacheSha256
    platform = 'windows-x86_64'
  })
  $SccacheRestored = Restore-ToolchainDependency 'sccache' $SccacheIdentity $SccacheDir
  if (-not (Test-Path -LiteralPath $script:WindowsTarExe)) {
    throw "Windows tar.exe was not found at $script:WindowsTarExe"
  }
  if (-not (Test-Path $SccacheExe)) {
    New-Item -ItemType Directory -Force -Path $SccacheDir | Out-Null
    $SccacheArchive = Join-Path $env:TEMP 'sccache.tar.gz'
    Invoke-WebRequestWithRetry `
      -Uri "https://github.com/mozilla/sccache/releases/download/$SccacheVersion/$SccacheFile.tar.gz" `
      -OutFile $SccacheArchive `
      -Description 'sccache archive'
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
  if (-not $SccacheRestored) {
    Publish-ToolchainDependency 'sccache' $SccacheIdentity $SccacheDir
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
    if ($Shard.build.cudaVersion -eq '13.1') {
      $CudaInstaller = Join-Path $ToolchainRoot 'cuda_13.1.2_windows_network.exe'
      $CudaInstallerUrl = 'https://developer.download.nvidia.com/compute/cuda/13.1.2/network_installers/cuda_13.1.2_windows_network.exe'
      $CudaInstallerMd5 = '2d5ebeee9c16f9fbe7186ac663bc0d58'
      Invoke-WebRequestWithRetry -Uri $CudaInstallerUrl -OutFile $CudaInstaller -Description 'CUDA 13.1.2 network installer'
      $ActualCudaInstallerMd5 = (Get-FileHash -LiteralPath $CudaInstaller -Algorithm MD5).Hash.ToLowerInvariant()
      if ($ActualCudaInstallerMd5 -ne $CudaInstallerMd5) {
        throw "CUDA 13.1.2 installer MD5 mismatch: expected $CudaInstallerMd5, got $ActualCudaInstallerMd5"
      }
      $CudaPackages = 'nvcc_13.1 visual_studio_integration_13.1 cublas_dev_13.1 cusolver_dev_13.1 curand_dev_13.1 nvrtc_dev_13.1 cudart_13.1 cusparse_dev_13.1'
      $CudaInstall = Start-Process -FilePath $CudaInstaller -ArgumentList "-s -n $CudaPackages" -Wait -PassThru
      if ($CudaInstall.ExitCode -ne 0) {
        throw "CUDA 13.1.2 installer failed with exit code $($CudaInstall.ExitCode)"
      }
    } else {
      $Installer = Join-Path $ToolchainRoot 'install_cuda_windows.ps1'
      Invoke-WebRequest 'https://raw.githubusercontent.com/KonduitAI/cuda-install/1bd33888dea7d372de612ec9ecc87343ec8dba4a/.github/actions/install-cuda-windows/install_cuda_windows.ps1' -OutFile $Installer -UseBasicParsing
      $PreviousGithubEnv = $env:GITHUB_ENV
      if ([string]::IsNullOrWhiteSpace($PreviousGithubEnv)) {
        $env:GITHUB_ENV = Join-Path $ToolchainRoot 'cuda-installer-github-env.txt'
      }
      try {
        & $Installer
      } finally {
        if ([string]::IsNullOrWhiteSpace($PreviousGithubEnv)) {
          Remove-Item Env:GITHUB_ENV -ErrorAction SilentlyContinue
        } else {
          $env:GITHUB_ENV = $PreviousGithubEnv
        }
      }
    }
    if (-not (Test-Path "$CudaPath\bin\nvcc.exe")) {
      throw "CUDA $($Shard.build.cudaVersion) installation did not provide nvcc.exe at $CudaPath"
    }
  }
  if ($Shard.build.cudaVersion -eq '13.1' -and -not (Test-Path "$CudaPath\include\cudnn.h")) {
    $CudnnVersion = '9.19.1.2'
    $CudnnSha256 = 'ffe9788ec702b8b0d26f43cf1fd6f099e312e62dd0b82e9793ff5ee21bd8e00a'
    $CudnnZip = Join-Path $env:TEMP "cudnn-$CudnnVersion-cuda13.zip"
    $CudnnDir = Join-Path $ToolchainRoot "cudnn-$CudnnVersion-cuda13"
    Invoke-WebRequestWithRetry -Uri "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-$($CudnnVersion)_cuda13-archive.zip" -OutFile $CudnnZip -Description "cuDNN $CudnnVersion CUDA 13 archive"
    $ActualCudnnSha256 = (Get-FileHash -LiteralPath $CudnnZip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualCudnnSha256 -ne $CudnnSha256) {
      throw "cuDNN $CudnnVersion archive SHA-256 mismatch: expected $CudnnSha256, got $ActualCudnnSha256"
    }
    Remove-Item -LiteralPath $CudnnDir -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -LiteralPath $CudnnZip -DestinationPath $CudnnDir -Force
    $CudnnRoot = Get-ChildItem -LiteralPath $CudnnDir -Directory | Select-Object -First 1
    if ($null -eq $CudnnRoot) { throw "cuDNN $CudnnVersion archive did not contain a redistribution root" }
    Copy-Item "$($CudnnRoot.FullName)\*" $CudaPath -Recurse -Force
  }
  $SparseSha256 = ''
  switch ($Shard.build.cudaVersion) {
    '12.6' { $SparseVersion = '12.5.4.2' }
    '12.9' { $SparseVersion = '12.5.10.65' }
    '13.1' {
      $SparseVersion = '12.7.3.1'
      $SparseSha256 = '602cf803627f75a2b123bbf7bf735389721274d0ad486697b43c1f1f74eb29cf'
    }
    default { throw "No cuSPARSE redistribution is pinned for CUDA $($Shard.build.cudaVersion)" }
  }
  if (-not (Test-Path "$CudaPath\include\cusparse_v2.h")) {
    $SparseZip = Join-Path $env:TEMP "cusparse-$($Shard.build.cudaVersion).zip"
    $SparseDir = Join-Path $ToolchainRoot "cusparse-$($Shard.build.cudaVersion)"
    Invoke-WebRequest "https://developer.download.nvidia.com/compute/cuda/redist/libcusparse/windows-x86_64/libcusparse-windows-x86_64-$SparseVersion-archive.zip" -OutFile $SparseZip -UseBasicParsing
    if ($SparseSha256) {
      $ActualSparseSha256 = (Get-FileHash -LiteralPath $SparseZip -Algorithm SHA256).Hash.ToLowerInvariant()
      if ($ActualSparseSha256 -ne $SparseSha256) {
        throw "cuSPARSE $SparseVersion archive SHA-256 mismatch: expected $SparseSha256, got $ActualSparseSha256"
      }
    }
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
    git -c core.autocrlf=false clone --filter=blob:none $Config.repository $SourceDir 2>&1
  }
  Invoke-NativeChecked -Description 'Source line-ending configuration' -Command {
    git -C $SourceDir config core.autocrlf false 2>&1
  }
  Invoke-NativeChecked -Description 'Pinned commit fetch' -Command {
    git -C $SourceDir fetch --depth=1 origin $Config.commit 2>&1
  }
  Invoke-NativeChecked -Description 'Pinned commit checkout' -Command {
    git -C $SourceDir checkout --detach $Config.commit 2>&1
  }
  $Actual = Invoke-NativeChecked -Description 'Pinned commit resolution' -Command {
    git -C $SourceDir rev-parse HEAD 2>&1
  }
  if ($Actual.Trim() -ne $Config.commit) { throw "Commit mismatch: $Actual" }
  # The controller embeds the local release script so a run uses the exact
  # release logic being tested even when the source commit predates it.
  $NativePlatformScript = Join-Path $SourceDir 'build-scripts/release/native-platform.sh'
  [IO.File]::WriteAllBytes($NativePlatformScript, [Convert]::FromBase64String($NativePlatformScriptB64))
  Write-Phase 'source-checkout' 'complete' "shard=$($Shard.id) commit=$($Config.commit) releaseScript=embedded"

  $MavenOutput = Join-Path $OutputDir 'maven-repository'
  $SdkOutput = Join-Path $OutputDir 'sdk-assets'
  New-Item -ItemType Directory -Force -Path $MavenOutput,$SdkOutput | Out-Null
  $Arguments = @($BuildDriver, '--config', $ShardConfigFile, '--source', $SourceDir, '--repository', $ShardMavenRepo, '--maven-output', $MavenOutput, '--sdk-output', $SdkOutput)
  Write-Phase 'matrix-build' 'started' "shard=$($Shard.id)"
  $Process = Start-Process $script:PythonExe -ArgumentList $Arguments -RedirectStandardOutput $MatrixLog -RedirectStandardError $MatrixError -PassThru -NoNewWindow
  $null = $Process.Handle
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
  $BuildExitCode = $Process.ExitCode
  Copy-NewLogContent $MatrixLog
  Copy-NewLogContent $MatrixError
  if ($null -eq $BuildExitCode) { throw 'Build process exited without an available exit code' }
  if ($BuildExitCode -eq 0) {
    Write-Phase 'matrix-build' 'complete' "shard=$($Shard.id)"
  }
  else {
    Write-Phase 'matrix-build' 'failed' "shard=$($Shard.id) exitCode=$BuildExitCode"
  }

  $HasMavenOutput = $null -ne (
    Get-ChildItem -LiteralPath $MavenOutput -Recurse -File -ErrorAction SilentlyContinue |
      Where-Object { $_.Extension -in @('.jar', '.pom') } |
      Select-Object -First 1
  )
  if ((@($Shard.workloads) -contains 'maven') -and ($BuildExitCode -eq 0 -or $HasMavenOutput)) {
    Write-Phase 'maven-publish' 'started' "shard=$($Shard.id) repository=$($Config.mavenRepositoryPrefix) buildExitCode=$BuildExitCode"
    $MavenAccounting = Publish-MavenRepository `
      $MavenOutput $SdkOutput $ShardConfigFile ($BuildExitCode -ne 0)
    Write-Phase 'maven-publish' 'complete' "shard=$($Shard.id) accounting=$MavenAccounting"
  }

  if ($BuildExitCode -ne 0) { throw "Build failed with exit code $BuildExitCode" }

  Write-Phase 'artifact-packaging' 'started' "shard=$($Shard.id)"
  Invoke-NativeChecked -Description 'SDK asset packaging' -Command {
    & $script:WindowsTarExe -C $SdkOutput -czf (Join-Path $OutputDir 'sdk-assets.tar.gz') .
  }
  # Windows PowerShell can corrupt nested quotes when a Python program is
  # passed through -c. Materialize the same streaming manifest writer used by
  # the Linux worker and execute the file instead.
  $ManifestScriptPath = Join-Path $BootstrapRoot 'write-shard-manifest.py'
  $ManifestScript = @'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    files.append({
        "path": path.relative_to(root).as_posix(),
        "sha256": digest.hexdigest(),
        "size": path.stat().st_size,
    })

with open(sys.argv[2], encoding="utf-8") as stream:
    config = json.load(stream)
shard = config["shard"]
with open(root / "shard-manifest.json", "w", encoding="utf-8") as stream:
    json.dump({
        "schemaVersion": 1,
        "provider": "azure",
        "runId": config["runId"],
        "shard": shard["id"],
        "commit": config["commit"],
        "releaseVersion": config["releaseVersion"],
        "workloads": shard["workloads"],
        "os": shard["os"],
        "platform": shard["build"]["javacppPlatform"],
        "backend": shard["build"]["backend"],
        "variants": [variant["name"] for variant in shard["build"]["variants"]],
        "files": files,
    }, stream, indent=2, sort_keys=True)
'@
  [IO.File]::WriteAllText($ManifestScriptPath, $ManifestScript)
  Invoke-NativeChecked -Description 'Shard manifest creation' -Command {
    & $script:PythonExe $ManifestScriptPath $OutputDir $ShardConfigFile
  }
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
  [IO.File]::WriteAllBytes($DependencyCache, [Convert]::FromBase64String($DependencyCacheB64))
  if ([string]::IsNullOrWhiteSpace($NativePlatformScriptB64)) { throw 'native-platform.sh payload is missing' }
  $Config = Get-Content -Raw $ConfigFile | ConvertFrom-Json
  $env:DL4J_CLOUD_IO = $CloudIo
  $env:DL4J_DEPENDENCY_CACHE_HELPER = $DependencyCache
  $env:AZURE_CLIENT_ID = $Config.managedIdentityClientId
  $Shards = if ($Config.shards) { @($Config.shards) } else { @($Config.shard) }
  if ($Shards.Count -eq 0) { throw 'Azure lane worker received no shards' }

  Set-ExecutionPolicy Bypass -Scope Process -Force
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Invoke-Expression ((New-Object Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
  }
  $PythonVersion = '3.12.10'
  $PythonInstall = Join-Path $env:SystemDrive 'Python312'
  $script:PythonExe = Join-Path $PythonInstall 'python.exe'
  $PythonInstaller = Join-Path $env:TEMP "python-$PythonVersion-amd64.exe"
  $PythonInstallerUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe"
  for ($PythonAttempt = 1; $PythonAttempt -le 3 -and -not (Test-Path -LiteralPath $script:PythonExe); $PythonAttempt++) {
    try {
      Remove-Item -LiteralPath $PythonInstaller -Force -ErrorAction SilentlyContinue
      Invoke-WebRequest -UseBasicParsing -Uri $PythonInstallerUrl -OutFile $PythonInstaller
      Invoke-NativeChecked -Description 'Python 3.12 installation' -SuccessCodes @(0) -Command {
        $PythonInstallerArguments = @('/quiet', 'InstallAllUsers=1', "TargetDir=$PythonInstall", 'Include_launcher=0', 'Include_test=0', 'PrependPath=0')
        $PythonInstallerProcess = Start-Process -FilePath $PythonInstaller -ArgumentList $PythonInstallerArguments -PassThru
        $null = $PythonInstallerProcess.Handle
        $PythonInstallerProcess.WaitForExit()
        $LASTEXITCODE = $PythonInstallerProcess.ExitCode
      }
    }
    catch {
      if ($PythonAttempt -ge 3) { throw }
      $PythonBackoff = 15 * $PythonAttempt
      Write-Warning "Python 3.12 direct installation attempt $PythonAttempt failed: $($_.Exception.Message)"
      Write-Output "[dl4j-phase] timestamp=$([DateTimeOffset]::UtcNow.ToString('o')) phase=python-runtime status=retrying attempt=$PythonAttempt backoffSeconds=$PythonBackoff"
      Start-Sleep -Seconds $PythonBackoff
    }
  }
  Remove-Item -LiteralPath $PythonInstaller -Force -ErrorAction SilentlyContinue
  $env:PATH = "${PythonInstall};${PythonInstall}\Scripts;C:\ProgramData\chocolatey\bin;$env:PATH"
  if (-not (Test-Path -LiteralPath $script:PythonExe)) {
    throw "Python 3.12 executable was not found at $script:PythonExe after direct installer completed"
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
