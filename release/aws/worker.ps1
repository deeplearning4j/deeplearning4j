$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$ConfigB64 = '__DL4J_WORKER_CONFIG_B64__'
$BuildDriverB64 = '__DL4J_BUILD_DRIVER_B64__'
$LogForwarderB64 = '__DL4J_LOG_FORWARDER_B64__'
$WorkRoot = 'C:\dl4j-release'
$ToolchainRoot = Join-Path $WorkRoot 'toolchains'
$env:CARGO_HOME = Join-Path $ToolchainRoot 'cargo'
$env:RUSTUP_HOME = Join-Path $ToolchainRoot 'rustup'
$env:TEMP = Join-Path $WorkRoot 'tmp'
$env:TMP = $env:TEMP
$env:PATH = "$($env:CARGO_HOME)\bin;$env:PATH"
$SourceDir = Join-Path $WorkRoot 'source'
$OutputDir = Join-Path $WorkRoot 'output'
$MavenRepo = Join-Path $WorkRoot 'm2'
$ConfigFile = Join-Path $WorkRoot 'worker.json'
$BuildDriver = Join-Path $WorkRoot 'build-platform.py'
$LogForwarder = Join-Path $WorkRoot 'log-forwarder.py'
$LogForwarderStop = Join-Path $WorkRoot 'log-forwarder.stop'
$LogForwarderError = Join-Path $WorkRoot 'log-forwarder.err'
$BootstrapLog = Join-Path $OutputDir 'bootstrap.log'
$BuildLog = Join-Path $OutputDir 'build.log'
New-Item -ItemType Directory -Force -Path $WorkRoot,$OutputDir,$MavenRepo,$env:CARGO_HOME,$env:RUSTUP_HOME,$env:TEMP | Out-Null
[IO.File]::WriteAllBytes($ConfigFile, [Convert]::FromBase64String($ConfigB64))
[IO.File]::WriteAllBytes($BuildDriver, [Convert]::FromBase64String($BuildDriverB64))
[IO.File]::WriteAllBytes($LogForwarder, [Convert]::FromBase64String($LogForwarderB64))
$Config = Get-Content -Raw $ConfigFile | ConvertFrom-Json
$Region = $Config.region
$Shard = $Config.shard
$Prefix = "s3://$($Config.bucket)/$($Config.artifactPrefix)/$($Config.runId)/$($Shard.id)"
$ExitCode = 1
$LogForwarderProcess = $null
$TranscriptStarted = $false

function Upload-IfPresent([string]$Path, [string]$Name) {
  if (Test-Path $Path) { aws --region $Region s3 cp $Path "$Prefix/$Name" --only-show-errors; if ($LASTEXITCODE -ne 0) { Write-Warning "Upload failed: $Name" } }
}

function Write-Phase([string]$Phase, [string]$Status, [string]$Detail = '') {
  $Message = "[dl4j-phase] timestamp=$([DateTimeOffset]::UtcNow.ToString('o')) phase=$Phase status=$Status $Detail"
  Write-Output $Message
  Add-Content -Path $BuildLog -Value $Message
}

function Start-LogForwarder {
  if ($LogForwarderProcess) { return }
  Write-Phase 'cloudwatch-forwarder' 'started'
  $LogArguments = @(
    $LogForwarder, '--file', $BuildLog, '--stop-file', $LogForwarderStop,
    '--region', $Region, '--group', $Config.logGroupName, '--stream', $Config.logStreamName
  )
  $script:LogForwarderProcess = Start-Process python -ArgumentList $LogArguments -RedirectStandardOutput $LogForwarderError -RedirectStandardError "$LogForwarderError.stderr" -PassThru -NoNewWindow
  Write-Phase 'cloudwatch-forwarder' 'complete' "pid=$($LogForwarderProcess.Id)"
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
  $VcVars = Join-Path $VsInstall 'VC\Auxiliary\Build\vcvars64.bat'
  if (-not (Test-Path -LiteralPath $VcVars)) {
    throw "Visual Studio x64 environment script was not found at $VcVars"
  }
  $EnvironmentLines = & $env:ComSpec /d /s /c "`"$VcVars`" >nul && set"
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
  Write-Phase 'visual-studio-environment' 'complete' "installation=$VsInstall"
}

Write-Phase 'worker' 'started' "pid=$PID"
try {
  Start-Transcript -Path $BootstrapLog -Append -Force | Out-Null
  $TranscriptStarted = $true
  Set-ExecutionPolicy Bypass -Scope Process -Force
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  Write-Phase 'chocolatey-bootstrap' 'started'
  if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Invoke-Expression ((New-Object Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
  }
  Write-Phase 'chocolatey-bootstrap' 'complete'
  Write-Phase 'logging-prerequisites' 'started'
  choco install -y --no-progress awscli python312
  $env:PATH = "C:\Program Files\Amazon\AWSCLIV2;C:\ProgramData\chocolatey\bin;$env:PATH"
  Write-Phase 'logging-prerequisites' 'complete'
  Start-LogForwarder
  Write-Phase 'toolchain-packages' 'started'
  choco install -y --no-progress cmake git maven ninja temurin11 7zip rustup.install visualstudio2022buildtools visualstudio2022-workload-vctools
  choco install -y --no-progress msys2 --params "/NoUpdate"
  Write-Phase 'toolchain-packages' 'complete'
  Import-VisualStudioEnvironment
  $env:PATH = "C:\Program Files\Git\cmd;C:\Program Files\Git\bin;C:\tools\msys64\mingw64\bin;C:\tools\msys64\usr\bin;$env:PATH"
  $JavaHome = Get-ChildItem 'C:\Program Files\Eclipse Adoptium' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
  if (-not $JavaHome) { $JavaHome = Get-ChildItem 'C:\Program Files\Java' -Directory | Sort-Object Name -Descending | Select-Object -First 1 }
  $env:JAVA_HOME = $JavaHome.FullName
  $env:PATH = "$env:JAVA_HOME\bin;$env:PATH"
  Write-Phase 'msys-toolchain' 'started'
  & C:\tools\msys64\usr\bin\bash.exe -lc "pacman-key --init && pacman-key --populate msys2 && pacman -S --needed --noconfirm base-devel git tar pkg-config unzip p7zip zip autoconf autoconf-archive automake patch make diffutils grep gzip mingw-w64-x86_64-make mingw-w64-x86_64-gnupg mingw-w64-x86_64-cmake mingw-w64-x86_64-nasm mingw-w64-x86_64-toolchain mingw-w64-x86_64-libtool mingw-w64-x86_64-gcc mingw-w64-x86_64-gcc-fortran mingw-w64-x86_64-libwinpthread-git mingw-w64-x86_64-SDL2 mingw-w64-x86_64-ragel mingw-w64-x86_64-sed mingw-w64-x86_64-ninja mingw-w64-x86_64-vulkan-headers mingw-w64-x86_64-vulkan-loader"
  Write-Phase 'msys-toolchain' 'complete'
  Write-Phase 'rust-toolchain' 'started'
  rustup toolchain install stable-x86_64-pc-windows-gnu
  rustup default stable-x86_64-pc-windows-gnu
  $env:CARGO_BUILD_TARGET = 'x86_64-pc-windows-gnu'
  cargo install --locked cbindgen
  if ($LASTEXITCODE -ne 0) { throw 'cbindgen installation failed' }
  Write-Phase 'rust-toolchain' 'complete'
  $SccacheVersion = 'v0.17.0'
  $SccacheFile = "sccache-$SccacheVersion-x86_64-pc-windows-msvc"
  $SccacheDir = Join-Path $ToolchainRoot 'sccache'
  New-Item -ItemType Directory -Force -Path $SccacheDir | Out-Null
  Invoke-WebRequest "https://github.com/mozilla/sccache/releases/download/$SccacheVersion/$SccacheFile.tar.gz" -OutFile (Join-Path $env:TEMP 'sccache.tar.gz') -UseBasicParsing
  tar -xzf (Join-Path $env:TEMP 'sccache.tar.gz') -C $env:TEMP
  Copy-Item (Join-Path $env:TEMP "$SccacheFile\sccache.exe") (Join-Path $SccacheDir 'sccache.exe') -Force
  $env:PATH = "$SccacheDir;$env:PATH"
  $env:SCCACHE_DIR = Join-Path $WorkRoot 'sccache'
  $env:SCCACHE_CACHE_SIZE = '100G'
  $env:SCCACHE_IDLE_TIMEOUT = '0'
  if ($Shard.build.backend -eq 'cuda') {
    Write-Phase 'cuda-toolchain' 'started' "version=$($Shard.build.cudaVersion)"
    $env:CUDA_VERSION = $Shard.build.cudaVersion
    $Installer = Join-Path $WorkRoot 'install_cuda_windows.ps1'
    Invoke-WebRequest 'https://raw.githubusercontent.com/KonduitAI/cuda-install/master/.github/actions/install-cuda-windows/install_cuda_windows.ps1' -OutFile $Installer -UseBasicParsing
    & $Installer
    $CudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$($Shard.build.cudaVersion)"
    $SparseVersion = if ($Shard.build.cudaVersion -eq '12.9') { '12.5.10.65' } else { '12.5.4.2' }
    $SparseZip = Join-Path $WorkRoot 'cusparse.zip'
    $SparseDir = Join-Path $WorkRoot 'cusparse'
    Invoke-WebRequest "https://developer.download.nvidia.com/compute/cuda/redist/libcusparse/windows-x86_64/libcusparse-windows-x86_64-$SparseVersion-archive.zip" -OutFile $SparseZip -UseBasicParsing
    Expand-Archive $SparseZip $SparseDir -Force
    $SparseRoot = Get-ChildItem $SparseDir -Directory | Select-Object -First 1
    Copy-Item "$($SparseRoot.FullName)\include\*" "$CudaPath\include\" -Recurse -Force
    Copy-Item "$($SparseRoot.FullName)\lib\x64\*" "$CudaPath\lib\x64\" -Recurse -Force
    if (Test-Path "$($SparseRoot.FullName)\bin") { Copy-Item "$($SparseRoot.FullName)\bin\*" "$CudaPath\bin\" -Recurse -Force }
    if (-not (Test-Path "$CudaPath\include\cusparse_v2.h")) { throw 'cuSPARSE installation is incomplete' }
    $env:CUDA_PATH = $CudaPath
    $env:CUDNN_ROOT_DIR = $CudaPath
    $env:PATH = "$CudaPath\bin;$CudaPath\libnvvp;$env:PATH"
    Write-Phase 'cuda-toolchain' 'complete' "version=$($Shard.build.cudaVersion)"
  }

  $Kill = aws --region $Region ssm get-parameter --name $Config.killSwitchParameter --query 'Parameter.Value' --output text
  if ($LASTEXITCODE -ne 0 -or $Kill -eq 'true') { throw 'Global kill switch is enabled or unreadable' }

  Write-Phase 'source-checkout' 'started' "commit=$($Config.commit)"
  git clone --filter=blob:none $Config.repository $SourceDir
  git -C $SourceDir fetch --depth=1 origin $Config.commit
  git -C $SourceDir checkout --detach $Config.commit
  $Actual = git -C $SourceDir rev-parse HEAD
  if ($Actual.Trim() -ne $Config.commit) { throw "Commit mismatch: $Actual" }
  Write-Phase 'source-checkout' 'complete' "commit=$($Config.commit)"

  $MavenOutput = Join-Path $OutputDir 'maven-repository'
  $SdkOutput = Join-Path $OutputDir 'sdk-assets'
  New-Item -ItemType Directory -Force -Path $MavenOutput,$SdkOutput | Out-Null
  Stop-Transcript | Out-Null
  $TranscriptStarted = $false
  Get-Content $BootstrapLog | Add-Content $BuildLog
  $Arguments = @(
    $BuildDriver, '--config', $ConfigFile,
    '--source', $SourceDir, '--repository', $MavenRepo,
    '--maven-output', $MavenOutput, '--sdk-output', $SdkOutput
  )
  Write-Phase 'matrix-build' 'started'
  $Process = Start-Process python -ArgumentList $Arguments -RedirectStandardOutput $BuildLog -RedirectStandardError "$BuildLog.err" -PassThru -NoNewWindow
  while (-not $Process.HasExited) {
    Start-Sleep -Seconds 15
    $Kill = aws --region $Region ssm get-parameter --name $Config.killSwitchParameter --query 'Parameter.Value' --output text 2>$null
    if ($LASTEXITCODE -ne 0 -or $Kill -eq 'true') {
      taskkill /PID $Process.Id /T /F | Out-Null
      shutdown.exe /s /t 0 /f
      throw 'Global kill switch stopped the build'
    }
    $Process.Refresh()
  }
  if ($Process.ExitCode -ne 0) { throw "Build failed with exit code $($Process.ExitCode)" }
  if (Test-Path "$BuildLog.err") { Get-Content "$BuildLog.err" | Add-Content $BuildLog }
  Write-Phase 'matrix-build' 'complete'
  Write-Phase 'artifact-packaging' 'started'

  python -c "import hashlib,json,pathlib,sys; root=pathlib.Path(sys.argv[1]); c=json.load(open(sys.argv[2])); files=[]; [(lambda p: files.append({'path':p.relative_to(root).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size':p.stat().st_size}))(p) for p in sorted(root.rglob('*')) if p.is_file()]; json.dump({'schemaVersion':1,'runId':c['runId'],'shard':c['shard']['id'],'commit':c['commit'],'releaseVersion':c['releaseVersion'],'workloads':c['shard']['workloads'],'os':c['shard']['os'],'platform':c['shard']['build']['javacppPlatform'],'backend':c['shard']['build']['backend'],'files':files},open(root/'shard-manifest.json','w'),indent=2,sort_keys=True)" $OutputDir $ConfigFile
  tar -C $MavenOutput -czf (Join-Path $OutputDir 'maven-repository.tar.gz') .
  tar -C $SdkOutput -czf (Join-Path $OutputDir 'sdk-assets.tar.gz') .
  Write-Phase 'artifact-packaging' 'complete'
  $ExitCode = 0
}
catch {
  Write-Phase 'worker' 'failed' $_.Exception.Message
  $_ | Out-String | Tee-Object -FilePath $BuildLog -Append
  $ExitCode = 1
}
finally {
  Write-Phase 'finalize' 'started' "exitCode=$ExitCode"
  if ($TranscriptStarted) {
    Stop-Transcript | Out-Null
    Get-Content $BootstrapLog | Add-Content $BuildLog
    $TranscriptStarted = $false
  }
  New-Item -ItemType File -Force -Path $LogForwarderStop | Out-Null
  if ($LogForwarderProcess) {
    Wait-Process -Id $LogForwarderProcess.Id -Timeout 40 -ErrorAction SilentlyContinue
    $LogForwarderProcess.Refresh()
    if (-not $LogForwarderProcess.HasExited) { Stop-Process -Id $LogForwarderProcess.Id -Force -ErrorAction SilentlyContinue }
  }
  if (Test-Path $LogForwarderError) { Get-Content $LogForwarderError | Add-Content $BuildLog }
  if (Test-Path "$LogForwarderError.stderr") { Get-Content "$LogForwarderError.stderr" | Add-Content $BuildLog }
  @{shard=$Shard.id; exitCode=$ExitCode; completedAt=[DateTimeOffset]::UtcNow.ToUnixTimeSeconds()} | ConvertTo-Json | Set-Content (Join-Path $OutputDir 'status.json')
  Upload-IfPresent $BuildLog 'build.log'
  Upload-IfPresent (Join-Path $OutputDir 'status.json') 'status.json'
  Upload-IfPresent (Join-Path $OutputDir 'maven-repository.tar.gz') 'maven-repository.tar.gz'
  Upload-IfPresent (Join-Path $OutputDir 'sdk-assets.tar.gz') 'sdk-assets.tar.gz'
  Upload-IfPresent (Join-Path $OutputDir 'shard-manifest.json') 'shard-manifest.json'
  shutdown.exe /s /t 0 /f
}
exit $ExitCode
