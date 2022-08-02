#!/bin/bash
env
## Based on the javacpp presets github actions centos 7 build at https://github.com/bytedeco/javacpp-presets/
SCL_ENABLE="devtoolset-7"
yum -y update && yum -y install wget unzip centos-release-scl-rh epel-release
echo "Downloading java from azul"
cd /opt && wget https://cdn.azul.com/zulu/bin/zulu11.58.15-ca-jdk11.0.16-linux_x64.zip
echo "Downloaded azul java"
ls /opt
cd /opt && unzip zulu11.58.15-ca-jdk11.0.16-linux_x64.zip
#zulu11.58.15-ca-jdk11.0.16-linux_x64
yum -y install $SCL_ENABLE rh-java-common-ant boost-devel ccache clang gcc-c++ gcc-gfortran  ant python python36-devel python36-pip swig file which wget unzip tar bzip2 gzip xz patch autoconf-archive automake make libtool bison flex perl nasm alsa-lib-devel freeglut-devel gtk2-devel libusb-devel libusb1-devel curl-devel expat-devel gettext-devel openssl-devel bzip2-devel zlib-devel SDL-devel libva-devel libxkbcommon-devel libxkbcommon-x11-devel xcb-util* fontconfig-devel libffi-devel ragel ocl-icd-devel GeoIP-devel pcre-devel ssdeep-devel yajl-devel
sed -i 's/_mm512_abs_pd (__m512 __A)/_mm512_abs_pd (__m512d __A)/g' /opt/rh/devtoolset-7/root/usr/lib/gcc/x86_64-redhat-linux/7/include/avx512fintrin.h
source scl_source enable $SCL_ENABLE || true
curl -LO https://github.com/Kitware/CMake/releases/download/v3.16.6/cmake-3.16.6-Linux-x86_64.tar.gz
curl -LO https://downloads.apache.org/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz
curl -LO https://mirrors.edge.kernel.org/pub/software/scm/git/git-2.18.3.tar.gz
tar -xzf cmake-3.16.6-Linux-x86_64.tar.gz -C /opt/
mv /opt/cmake-3.16.6-Linux-x86_64 /opt/cmake
tar -xzf apache-maven-3.6.3-bin.tar.gz -C /opt/
tar -xzf git-2.18.3.tar.gz
pushd git-2.18.3; make -j2 prefix=/usr/local/; make -j2 prefix=/usr/local/ install; popd
ln -sf /usr/bin/python3.6 /usr/bin/python3
ln -sf /opt/cmake-3.16.6-Linux-x86_64/bin/* /usr/bin/
ln -sf /opt/apache-maven-3.6.3/bin/mvn /usr/bin/mvn
curl -fsSL https://github.com/google/protobuf/releases/download/v3.8.0/protobuf-cpp-3.8.0.tar.gz \
                       | tar xz && \
                       cd protobuf-3.8.0 && \
                       ./configure --prefix=/opt/protobuf && \
                       make -j2 && \
                       make install && \
                       cd .. && \
                       rm -rf protobuf-3.8.0
echo "/opt/protobuf/bin" >> $GITHUB_PATH
# need to hardcode due to conflicting java home being set
export JAVA_HOME=/opt/zulu11.58.15-ca-jdk11.0.16-linux_x64
echo "${JAVA_HOME}/bin" >> $GITHUB_PATH
export PATH=/opt/protobuf/bin:/opt/cmake/bin:$JAVA_HOME/bin:$PATH
echo "JAVA_HOME ${JAVA_HOME}"
java -version
which javac
mvn --version
cmake --version
protoc --version
pwd
# The volume directory for the workspace
cd "/github/workspace/"
bash ./bootstrap-libnd4j-from-url.sh

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$OPENBLAS_PATH"
echo "Running INSTALL COMMAND ${INSTALL_COMMAND}"
eval "${INSTALL_COMMAND}"

