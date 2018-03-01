call "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat" amd64
echo on

if "%APPVEYOR_PULL_REQUEST_NUMBER%" == "" (
    set BRANCH=%APPVEYOR_REPO_BRANCH%
    set MAVEN_PHASE=deploy
) else (
    set BRANCH=%APPVEYOR_PULL_REQUEST_HEAD_REPO_BRANCH%
    set MAVEN_PHASE=install
)

git -C "%APPVEYOR_BUILD_FOLDER%\.." clone https://github.com/deeplearning4j/libnd4j/ --depth=50 --branch=%BRANCH%
if %ERRORLEVEL% neq 0 (
    git -C "%APPVEYOR_BUILD_FOLDER%\.." clone https://github.com/deeplearning4j/libnd4j/ --depth=50
)

if "%CUDA%" == "8.0" (
    set "CUDA_URL=https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_windows-exe"
)
if "%CUDA%" == "9.0" (
    set "CUDA_URL=https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_windows-exe"
)
if "%CUDA%" == "9.1" (
    set "CUDA_URL=https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_windows"
)
if not "%CUDA_URL%" == "" (
    curl --retry 10 -L -o cuda.exe %CUDA_URL%
    cuda.exe -s
    set "CUDA_PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA%"
    set "CUDA_PATH_V%CUDA:.=_%=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA%"
    set "PATH=%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA%\bin;%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA%\libnvvp;%PATH%"
)

set "PATH=C:\msys64\usr\bin\core_perl;C:\msys64\mingw64\bin;C:\msys64\usr\bin;%PATH%"
bash -lc "pacman -Syu --noconfirm"
bash -lc "pacman -Su --noconfirm"
bash -lc "pacman -S --needed --noconfirm base-devel make mingw-w64-x86_64-cmake mingw-w64-x86_64-gcc"

if not "%CUDA%" == "" (
    bash -c "cd ../libnd4j/; MAKEJ=1 bash buildnativeoperations.sh -c cuda -v $CUDA -cc 30"
    bash -c "bash change-cuda-versions.sh $CUDA"
    set "EXTRA_OPTIONS=-pl !nd4j-uberjar,!nd4j-backends/nd4j-backend-impls/nd4j-native,!nd4j-backends/nd4j-backend-impls/nd4j-native-platform,!nd4j-backends/nd4j-tests"
) else (
    bash -c "cd ../libnd4j/; MAKEJ=2 bash buildnativeoperations.sh -c cpu -e $EXT"
    set "EXTRA_OPTIONS=-pl !nd4j-uberjar,!nd4j-backends/nd4j-backend-impls/nd4j-cuda,!nd4j-backends/nd4j-backend-impls/nd4j-cuda-platform,!nd4j-backends/nd4j-tests"
)
bash -c "bash change-scala-versions.sh $SCALA"
call mvn clean %MAVEN_PHASE% -B -U --settings .\ci\settings.xml -Dmaven.test.skip=true -Dlocal.software.repository=sonatype ^
    -Djavacpp.extension=%EXT% %EXTRA_OPTIONS%
