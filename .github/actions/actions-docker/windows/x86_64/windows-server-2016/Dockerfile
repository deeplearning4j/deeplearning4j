# https://github.com/JonCubed/docker-jenkins-jnlp-slave-windows
# https://github.com/StefanScherer/dockerfiles-windows

# FROM jonathank/jenkins-jnlp-slave-windows

FROM microsoft/windowsservercore:ltsc2019 AS builder

WORKDIR c:/windows/temp

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Install JDK
ENV JAVA_VERSION 8.19.0.1-jdk8.0.112
ENV JAVA_MD5 70bbe971a9a6995a9e909b709ae52b3
ENV JAVA_HOME C:\\zulu${JAVA_VERSION}-win_x64

RUN (New-Object System.Net.WebClient).DownloadFile('https://cdn.azul.com/zulu/bin/zulu{0}-win_x64.zip' -f $env:JAVA_VERSION, 'openjdk.zip') ; \
    if ((Get-FileHash openjdk.zip -Algorithm md5).Hash -ne $env:JAVA_MD5) {exit 1} ; \
    Expand-Archive openjdk.zip -DestinationPath C:\ ; \
    $env:PATH = '{0}\bin;{1}' -f $env:JAVA_HOME, $env:PATH ; \
    [Environment]::SetEnvironmentVariable('PATH', $env:PATH, [EnvironmentVariableTarget]::Machine) ; \
    Remove-Item -Path openjdk.zip

# Install MSYS2
RUN powershell -Command \
    $ProgressPreference = 'SilentlyContinue' ; \
    Write-Output 'Downloading msys2' ; \
    curl.exe -o msys2.tar.xy http://repo.msys2.org/distrib/msys2-x86_64-latest.tar.xz ; \
    Install-Package -Scope CurrentUser -Force 7Zip4Powershell ; \
    Write-Output 'Extracting tar.xz' ; \
    Expand-7zip msys2.tar.xy . ; \
    Write-Output 'Extracting tar' ; \
    Expand-7zip msys2.tar C:/ ; \
    Write-Output 'Done'

# Install MSYS2 pacakges
RUN C:\msys64\usr\bin\pacman.exe -Syu
RUN C:\msys64\usr\bin\pacman.exe -Su
RUN C:\msys64\usr\bin\pacman.exe -S \
        mingw-w64-x86_64-gcc \
        mingw-w64-x86_64-cmake \
        mingw-w64-x86_64-extra-cmake-modules \
        make \
        pkg-config \
        grep \
        sed \
        gzip \
        tar \
        mingw64/mingw-w64-x86_64-openblas \
        mingw-w64-x86_64-lz4 \
        mingw-w64-x86_64-gdb \
        mingw-w64-x86_64-make \
        mingw-w64-x86_64-ninja

FROM microsoft/windowsservercore:ltsc2019
COPY --from=builder C:/Windows/system32/netapi32.dll C:/Windows/system32/netapi32.dll
COPY --from=builder C:/msys64 C:/msys64
COPY --from=builder C:/zulu8.19.0.1-jdk8.0.112-win_x64 C:/zulu8.19.0.1-jdk8.0.112-win_x64

#ENV PATH C:\msys64\usr\bin;C:\Windows;C:\Windows\System32
ENV MSYSTEM MSYS2

CMD powershell