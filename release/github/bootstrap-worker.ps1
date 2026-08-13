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
$msysPath = 'C:\msys64\usr\bin'
$mingwPath = 'C:\msys64\mingw64\bin'
$gitBash = 'C:\Program Files\Git\bin\bash.exe'
if (-not (Test-Path -LiteralPath $gitBash)) {
  throw "Git Bash was not installed at $gitBash"
}
Add-Content -Path $env:GITHUB_PATH -Value $msysPath
Add-Content -Path $env:GITHUB_PATH -Value $mingwPath
Add-Content -Path $env:GITHUB_ENV -Value "DL4J_BASH_EXE=$gitBash"

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

if ($Shard -match 'cuda-12-([69])' -or $Shard -match 'zluda') {
  $cudaVersion = if ($Shard -match '12-6') { '12.6' } else { '12.9' }
  $installer = Join-Path $env:RUNNER_TEMP 'install_cuda_windows.ps1'
  Invoke-WebRequest 'https://raw.githubusercontent.com/KonduitAI/cuda-install/1bd33888dea7d372de612ec9ecc87343ec8dba4a/.github/actions/install-cuda-windows/install_cuda_windows.ps1' -OutFile $installer -UseBasicParsing
  $env:CUDA_VERSION = $cudaVersion
  & $installer

  $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$cudaVersion"
  if (-not (Test-Path -LiteralPath (Join-Path $cudaPath 'bin\nvcc.exe'))) {
    throw "CUDA bootstrap did not create nvcc.exe under $cudaPath"
  }

  # The upstream installer omits cuSPARSE. libnd4j includes cusparse_v2.h and
  # links its import library, so install the version from the matching CUDA
  # redistribution manifest into the same toolkit root.
  $sparseVersion = if ($cudaVersion -eq '12.9') { '12.5.10.65' } else { '12.5.4.2' }
  if (-not (Test-Path -LiteralPath (Join-Path $cudaPath 'include\cusparse_v2.h'))) {
    $sparseZip = Join-Path $env:RUNNER_TEMP "cusparse-$cudaVersion.zip"
    $sparseDir = Join-Path $env:RUNNER_TEMP "cusparse-$cudaVersion"
    Invoke-WebRequest "https://developer.download.nvidia.com/compute/cuda/redist/libcusparse/windows-x86_64/libcusparse-windows-x86_64-$sparseVersion-archive.zip" -OutFile $sparseZip -UseBasicParsing
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
  Add-Content -Path $env:GITHUB_ENV -Value 'CUDAFLAGS=--allow-unsupported-compiler'
  Add-Content -Path $env:GITHUB_ENV -Value 'NVCC_APPEND_FLAGS=--allow-unsupported-compiler'
  Add-Content -Path $env:GITHUB_ENV -Value 'CMAKE_CUDA_FLAGS=--allow-unsupported-compiler'
}

Write-Host "[dl4j-bootstrap] shard=$Shard status=complete"
