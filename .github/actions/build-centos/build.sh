#!/bin/bash
env
## Compat build using AlmaLinux 8 (glibc 2.28, GCC 8.5)
## Provides manylinux_2_28 compatibility — works on Ubuntu 20.04+, Debian 10+, RHEL 8+, Fedora 28+

# Enable PowerTools repo (needed for some -devel packages on AlmaLinux 8)
dnf -y install dnf-plugins-core epel-release
dnf config-manager --set-enabled powertools

# Install development tools and dependencies
dnf -y groupinstall "Development Tools"
dnf -y install wget unzip ccache clang gcc-c++ gcc-gfortran \
    python3 python3-devel python3-pip swig file which tar bzip2 gzip xz \
    patch autoconf automake make libtool bison flex perl nasm \
    alsa-lib-devel freeglut-devel gtk3-devel libusb-devel libusbx-devel \
    curl-devel expat-devel gettext-devel openssl-devel bzip2-devel zlib-devel \
    SDL2-devel libva-devel libxkbcommon-devel libxkbcommon-x11-devel \
    fontconfig-devel libffi-devel pcre-devel git cmake \
    openblas-devel

echo "Downloading java from azul"
cd /opt && wget https://cdn.azul.com/zulu/bin/zulu11.58.15-ca-jdk11.0.16-linux_x64.zip
echo "Downloaded azul java"
ls /opt
cd /opt && unzip zulu11.58.15-ca-jdk11.0.16-linux_x64.zip

curl -LO https://archive.apache.org/dist/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz
tar -xzf apache-maven-3.6.3-bin.tar.gz -C /opt/

curl -fsSL https://github.com/google/protobuf/releases/download/v3.8.0/protobuf-cpp-3.8.0.tar.gz \
                       | tar xz && \
                       cd protobuf-3.8.0 && \
                       ./configure --prefix=/opt/protobuf && \
                       make -j2 && \
                       make install && \
                       cd .. && \
                       rm -rf protobuf-3.8.0

ln -sf /opt/apache-maven-3.6.3/bin/mvn /usr/bin/mvn
echo "/opt/protobuf/bin" >> $GITHUB_PATH

# need to hardcode due to conflicting java home being set
export JAVA_HOME=/opt/zulu11.58.15-ca-jdk11.0.16-linux_x64
echo "${JAVA_HOME}/bin" >> $GITHUB_PATH
export PATH=/opt/protobuf/bin:$JAVA_HOME/bin:$PATH
echo "JAVA_HOME ${JAVA_HOME}"
java -version
which javac
mvn --version
cmake --version
gcc --version
protoc --version
pwd
# The volume directory for the workspace
cd "/github/workspace/"
bash ./bootstrap-libnd4j-from-url.sh

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$OPENBLAS_PATH"
echo "Running INSTALL COMMAND ${INSTALL_COMMAND}"
eval "${INSTALL_COMMAND}"
