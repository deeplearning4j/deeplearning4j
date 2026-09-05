param(
  [Parameter(Mandatory = $true)]
  [string]$Shard
)

$ErrorActionPreference = 'Stop'
Set-ExecutionPolicy Bypass -Scope Process -Force

choco install -y --no-progress cmake git maven ninja temurin11 7zip
rustup toolchain install stable-x86_64-pc-windows-gnu
rustup default stable-x86_64-pc-windows-gnu
if (-not (Get-Command cbindgen -ErrorAction SilentlyContinue)) {
  cargo install --locked cbindgen
}
$gitBash = 'C:\Program Files\Git\bin\bash.exe'
if (-not (Test-Path -LiteralPath $gitBash)) {
  throw "Git Bash was not installed at $gitBash"
}
# The MSYS2 setup action installs the pinned MINGW64 toolchain before this
# script runs. Prefer that root explicitly; falling back to PATH is allowed
# for older callers, but never mix gcc and g++ from different installations.
$preferredMingwPath = 'C:\msys64\mingw64\bin'
$preferredGcc = Join-Path $preferredMingwPath 'gcc.exe'
$gccCommand = Get-Command gcc.exe -ErrorAction SilentlyContinue
$gccPath = if (Test-Path -LiteralPath $preferredGcc) {
  $preferredGcc
} elseif ($gccCommand) {
  $gccCommand.Source
} else {
  $null
}
if ([string]::IsNullOrWhiteSpace($gccPath)) {
  throw 'Pinned MSYS2 MinGW64 gcc.exe was not found; toolchain setup did not complete'
}
$mingwPath = Split-Path -Parent $gccPath
$gxxPath = Join-Path $mingwPath 'g++.exe'
if (-not (Test-Path -LiteralPath $gxxPath)) {
  throw "MinGW g++.exe is missing beside gcc.exe at $mingwPath"
}
$mingwTarget = (& $gccPath -dumpmachine | Out-String).Trim()
if ($mingwTarget -notmatch '^x86_64-w64-mingw32') {
  throw "Expected an x86_64 MinGW compiler, got '$mingwTarget' from $gccPath"
}
$runtimeNames = @('libstdc++-6.dll', 'libgcc_s_seh-1.dll', 'libwinpthread-1.dll')
$missingRuntime = @($runtimeNames | Where-Object { -not (Test-Path -LiteralPath (Join-Path $mingwPath $_)) })
if ($missingRuntime.Count -gt 0) {
  throw "MinGW runtime is incomplete under ${mingwPath}: $($missingRuntime -join ', ')"
}
$msysRoot = Split-Path -Parent (Split-Path -Parent $mingwPath)
$msysPath = Join-Path $msysRoot 'usr\bin'
if (Test-Path -LiteralPath $msysPath) {
  Add-Content -Path $env:GITHUB_PATH -Value $msysPath
}
Add-Content -Path $env:GITHUB_PATH -Value $mingwPath
Add-Content -Path $env:GITHUB_ENV -Value "DL4J_BASH_EXE=$gitBash"
Add-Content -Path $env:GITHUB_ENV -Value "DL4J_MINGW_BIN=$mingwPath"
Add-Content -Path $env:GITHUB_ENV -Value "DL4J_MINGW_GCC=$gccPath"
Add-Content -Path $env:GITHUB_ENV -Value "DL4J_MINGW_GXX=$gxxPath"
Write-Host "[dl4j-bootstrap] mingw-target=$mingwTarget gcc=$gccPath gxx=$gxxPath"

# Use the exact upstream host compiler for schema generation. The MinGW-built
# flatc from the same source tree has crashed while executing on hosted Windows
# runners; the official binary avoids making schema generation depend on that
# host compiler/runtime combination.
$flatbuffersVersion = '25.2.10'
$flatcZip = Join-Path $env:RUNNER_TEMP "flatc-$flatbuffersVersion.zip"
$flatcDir = Join-Path $env:RUNNER_TEMP "flatc-$flatbuffersVersion"
Invoke-WebRequest "https://github.com/google/flatbuffers/releases/download/v$flatbuffersVersion/Windows.flatc.binary.zip" -OutFile $flatcZip -UseBasicParsing
Remove-Item -LiteralPath $flatcDir -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -LiteralPath $flatcZip -DestinationPath $flatcDir -Force
$flatc = Get-ChildItem -LiteralPath $flatcDir -Filter flatc.exe -File -Recurse | Select-Object -First 1
if ($null -eq $flatc) {
  throw "FlatBuffers $flatbuffersVersion archive did not contain flatc.exe"
}
$flatcVersion = (& $flatc.FullName --version | Out-String).Trim()
if ($flatcVersion -notmatch [regex]::Escape($flatbuffersVersion)) {
  throw "Expected flatc $flatbuffersVersion, got '$flatcVersion'"
}
$flatcPath = $flatc.FullName.Replace('\', '/')
Add-Content -Path $env:GITHUB_ENV -Value "DL4J_FLATC_EXECUTABLE=$flatcPath"
Write-Host "[dl4j-bootstrap] flatc=$flatcVersion path=$flatcPath"

# JavaCPP's Windows SDX builder uses MSVC's cl.exe. GitHub-hosted Windows
# runners have Visual Studio installed, but its compiler environment is not
# active in a normal PowerShell/Git Bash step, so materialize vcvars64 for this
# worker and persist the required variables for the later shared-worker step.
  $vsWhereRoot = ${env:ProgramFiles(x86)}
  if (-not $vsWhereRoot) { $vsWhereRoot = $env:ProgramFiles }
  $vsWhere = Join-Path $vsWhereRoot 'Microsoft Visual Studio\Installer\vswhere.exe'
  if (-not (Test-Path -LiteralPath $vsWhere)) {
    throw "Visual Studio locator was not found at $vsWhere"
  }
  $vsInstall = (& $vsWhere -latest -products '*' -version '[17.0,18.0)' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)
  if (-not $vsInstall) {
    throw 'Visual Studio 2022 C++ Build Tools installation was not found'
  }
  $vcVarsVersion = $null
  if ($Shard -match 'cuda-(12-6|13-1)') {
    # CUDA 12.6 and 13.1 support Visual Studio 2022 with the MSVC 193x
    # compiler family. Current hosted runners default to a newer 194x
    # compiler, so install and select a supported toolset explicitly.
    $cudaSupportedToolsetVersion = '14.38'
    $cudaSupportedToolsetComponent = 'Microsoft.VisualStudio.Component.VC.14.38.17.8.x86.x64'
    $toolsetRoot = Join-Path $vsInstall 'VC\Tools\MSVC'
    $cudaSupportedToolset = Get-ChildItem -LiteralPath $toolsetRoot -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -like "$cudaSupportedToolsetVersion.*" } |
      Sort-Object Name -Descending |
      Select-Object -First 1
    if ($null -eq $cudaSupportedToolset) {
      $vsSetup = Join-Path $vsWhereRoot 'Microsoft Visual Studio\Installer\setup.exe'
      if (-not (Test-Path -LiteralPath $vsSetup)) {
        throw "Visual Studio installer was not found at $vsSetup"
      }
      $setupArguments = @(
        'modify', '--installPath', "`"$vsInstall`"",
        '--channelId', 'VisualStudio.17.Release',
        '--productId', 'Microsoft.VisualStudio.Product.Enterprise',
        '--add', $cudaSupportedToolsetComponent,
        '--quiet', '--norestart'
      )
      $setup = Start-Process -FilePath $vsSetup -ArgumentList $setupArguments -Wait -PassThru
      if ($setup.ExitCode -notin @(0, 3010)) {
        throw "Installing $cudaSupportedToolsetComponent failed with exit code $($setup.ExitCode)"
      }
      $cudaSupportedToolset = Get-ChildItem -LiteralPath $toolsetRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "$cudaSupportedToolsetVersion.*" } |
        Sort-Object Name -Descending |
        Select-Object -First 1
    }
    if ($null -eq $cudaSupportedToolset) {
      throw "CUDA shard $Shard requires MSVC 193x, but toolset $cudaSupportedToolsetVersion was not installed"
    }
    $vcVarsVersion = $cudaSupportedToolsetVersion
  }
  $vcVars = Join-Path $vsInstall 'VC\Auxiliary\Build\vcvars64.bat'
  if (-not (Test-Path -LiteralPath $vcVars)) {
    throw "Visual Studio x64 environment script was not found at $vcVars"
  }
  $originalPathSet = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
  foreach ($pathEntry in ($env:PATH -split ';')) {
    if ($pathEntry) { [void]$originalPathSet.Add($pathEntry) }
  }
  $vcVarsCommand = "`"$vcVars`""
  if ($vcVarsVersion) {
    $vcVarsCommand += " -vcvars_ver=$vcVarsVersion"
  }
  $environmentLines = & $env:ComSpec /d /s /c "$vcVarsCommand >nul && set"
  if ($LASTEXITCODE -ne 0) {
    throw "Visual Studio x64 environment initialization failed with exit code $LASTEXITCODE"
  }
  foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') {
      [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
    }
  }
  if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'Visual Studio environment did not expose cl.exe on PATH'
  }
  if ($vcVarsVersion -and $env:VCToolsVersion -notlike "$vcVarsVersion.*") {
    throw "Requested MSVC $vcVarsVersion for CUDA shard $Shard, but vcvars selected $($env:VCToolsVersion)"
  }

  $vsEnvironmentNames = @(
    'CL', 'DevEnvDir', 'Framework40Version', 'FrameworkDir',
    'FrameworkDir32', 'FrameworkVersion', 'FrameworkVersion32', 'INCLUDE',
    'LIB', 'LIBPATH', 'NETFXSDKDir', 'UCRTVersion', 'UniversalCRTSdkDir',
    'VCIDEInstallDir', 'VCINSTALLDIR', 'VCToolsInstallDir',
    'VCToolsRedistDir', 'VCToolsVersion', 'VisualStudioVersion',
    'VS170COMNTOOLS', 'VSCMD_ARG_app_plat', 'VSCMD_ARG_HOST_ARCH',
    'VSCMD_ARG_TGT_ARCH', 'VSCMD_VER', 'VSINSTALLDIR', 'WindowsLibPath',
    'WindowsSdkBinPath', 'WindowsSdkDir', 'WindowsSDKLibVersion',
    'WindowsSdkVerBinPath', 'WindowsSDKVersion'
  )
  foreach ($name in $vsEnvironmentNames) {
    $value = [Environment]::GetEnvironmentVariable($name, 'Process')
    if ($null -ne $value) {
      Add-Content -Path $env:GITHUB_ENV -Value "$name=$value"
    }
  }
  # GITHUB_PATH augments the next step's existing PATH. Persisting the complete
  # vcvars PATH duplicates every hosted-runner entry and can make nvcc's own
  # vcvars64 invocation exceed cmd.exe's command-line limit. Persist only the
  # Visual Studio entries that vcvars added.
  foreach ($pathEntry in ($env:PATH -split ';')) {
    if ($pathEntry -and -not $originalPathSet.Contains($pathEntry)) {
      Add-Content -Path $env:GITHUB_PATH -Value $pathEntry
    }
  }
  Write-Host "[dl4j-bootstrap] visual-studio=$vsInstall cl=$((Get-Command cl.exe).Source)"

if ($Shard -match 'cuda-(12-([69])|13-1)' -or $Shard -match 'zluda') {
  $cudaVersion = if ($Shard -match '13-1') {
    '13.1'
  } elseif ($Shard -match '12-6') {
    '12.6'
  } else {
    '12.9'
  }
  $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$cudaVersion"
  if ($cudaVersion -eq '13.1') {
    $cudaInstaller = Join-Path $env:RUNNER_TEMP 'cuda_13.1.2_windows_network.exe'
    $cudaInstallerUrl = 'https://developer.download.nvidia.com/compute/cuda/13.1.2/network_installers/cuda_13.1.2_windows_network.exe'
    $cudaInstallerMd5 = '2d5ebeee9c16f9fbe7186ac663bc0d58'
    Invoke-WebRequest $cudaInstallerUrl -OutFile $cudaInstaller -UseBasicParsing
    $actualCudaInstallerMd5 = (Get-FileHash -LiteralPath $cudaInstaller -Algorithm MD5).Hash.ToLowerInvariant()
    if ($actualCudaInstallerMd5 -ne $cudaInstallerMd5) {
      throw "CUDA 13.1.2 installer MD5 mismatch: expected $cudaInstallerMd5, got $actualCudaInstallerMd5"
    }
    $cudaPackages = 'nvcc_13.1 visual_studio_integration_13.1 cublas_dev_13.1 cusolver_dev_13.1 curand_dev_13.1 nvrtc_dev_13.1 cudart_13.1 cusparse_dev_13.1'
    $cudaInstall = Start-Process -FilePath $cudaInstaller -ArgumentList "-s -n $cudaPackages" -Wait -PassThru
    if ($cudaInstall.ExitCode -ne 0) {
      throw "CUDA 13.1.2 installer failed with exit code $($cudaInstall.ExitCode)"
    }

    $cudnnVersion = '9.19.1.2'
    $cudnnSha256 = 'ffe9788ec702b8b0d26f43cf1fd6f099e312e62dd0b82e9793ff5ee21bd8e00a'
    $cudnnZip = Join-Path $env:RUNNER_TEMP "cudnn-$cudnnVersion-cuda13.zip"
    $cudnnDir = Join-Path $env:RUNNER_TEMP "cudnn-$cudnnVersion-cuda13"
    Invoke-WebRequest "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-$($cudnnVersion)_cuda13-archive.zip" -OutFile $cudnnZip -UseBasicParsing
    $actualCudnnSha256 = (Get-FileHash -LiteralPath $cudnnZip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualCudnnSha256 -ne $cudnnSha256) {
      throw "cuDNN $cudnnVersion archive SHA-256 mismatch: expected $cudnnSha256, got $actualCudnnSha256"
    }
    Remove-Item -LiteralPath $cudnnDir -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -LiteralPath $cudnnZip -DestinationPath $cudnnDir -Force
    $cudnnRoot = Get-ChildItem -LiteralPath $cudnnDir -Directory | Select-Object -First 1
    if ($null -eq $cudnnRoot) {
      throw "cuDNN $cudnnVersion archive did not contain a redistribution root"
    }
    Copy-Item "$($cudnnRoot.FullName)\*" $cudaPath -Recurse -Force
  } else {
    $installer = Join-Path $env:RUNNER_TEMP 'install_cuda_windows.ps1'
    Invoke-WebRequest 'https://raw.githubusercontent.com/KonduitAI/cuda-install/1bd33888dea7d372de612ec9ecc87343ec8dba4a/.github/actions/install-cuda-windows/install_cuda_windows.ps1' -OutFile $installer -UseBasicParsing
    $env:CUDA_VERSION = $cudaVersion
    & $installer
  }

  if (-not (Test-Path -LiteralPath (Join-Path $cudaPath 'bin\nvcc.exe'))) {
    throw "CUDA bootstrap did not create nvcc.exe under $cudaPath"
  }

  # The upstream installer omits cuSPARSE. libnd4j includes cusparse_v2.h and
  # links its import library, so install the version from the matching CUDA
  # redistribution manifest into the same toolkit root.
  $sparseSha256 = ''
  switch ($cudaVersion) {
    '12.6' { $sparseVersion = '12.5.4.2' }
    '12.9' { $sparseVersion = '12.5.10.65' }
    '13.1' {
      $sparseVersion = '12.7.3.1'
      $sparseSha256 = '602cf803627f75a2b123bbf7bf735389721274d0ad486697b43c1f1f74eb29cf'
    }
    default { throw "No cuSPARSE redistribution is pinned for CUDA $cudaVersion" }
  }
  if (-not (Test-Path -LiteralPath (Join-Path $cudaPath 'include\cusparse_v2.h'))) {
    $sparseZip = Join-Path $env:RUNNER_TEMP "cusparse-$cudaVersion.zip"
    $sparseDir = Join-Path $env:RUNNER_TEMP "cusparse-$cudaVersion"
    Invoke-WebRequest "https://developer.download.nvidia.com/compute/cuda/redist/libcusparse/windows-x86_64/libcusparse-windows-x86_64-$sparseVersion-archive.zip" -OutFile $sparseZip -UseBasicParsing
    if ($sparseSha256) {
      $actualSparseSha256 = (Get-FileHash -LiteralPath $sparseZip -Algorithm SHA256).Hash.ToLowerInvariant()
      if ($actualSparseSha256 -ne $sparseSha256) {
        throw "cuSPARSE $sparseVersion archive SHA-256 mismatch: expected $sparseSha256, got $actualSparseSha256"
      }
    }
    Remove-Item -LiteralPath $sparseDir -Recurse -Force -ErrorAction SilentlyContinue
    Expand-Archive -LiteralPath $sparseZip -DestinationPath $sparseDir -Force
    $sparseRoot = Get-ChildItem -LiteralPath $sparseDir -Directory | Select-Object -First 1
    if ($null -eq $sparseRoot) {
      throw "cuSPARSE $sparseVersion archive did not contain a redistribution root"
    }
    Copy-Item "$($sparseRoot.FullName)\include\*" "$cudaPath\include\" -Recurse -Force
    Copy-Item "$($sparseRoot.FullName)\lib\x64\*" "$cudaPath\lib\x64\" -Recurse -Force
    if (Test-Path -LiteralPath "$($sparseRoot.FullName)\bin") {
      Copy-Item "$($sparseRoot.FullName)\bin\*" "$cudaPath\bin\" -Recurse -Force
    }
  }
  if (-not (Test-Path -LiteralPath (Join-Path $cudaPath 'include\cusparse_v2.h'))) {
    throw "cuSPARSE $sparseVersion installation is incomplete under $cudaPath"
  }

  Add-Content -Path $env:GITHUB_ENV -Value "CUDA_PATH=$cudaPath"
  Add-Content -Path $env:GITHUB_ENV -Value "CUDNN_ROOT_DIR=$cudaPath"
  Add-Content -Path $env:GITHUB_PATH -Value (Join-Path $cudaPath 'bin')
}

Write-Host "[dl4j-bootstrap] shard=$Shard status=complete"
