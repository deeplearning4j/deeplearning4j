ARG BASEIMAGE_VERSION=7
FROM amd64/centos:${BASEIMAGE_VERSION} AS base_image

FROM base_image AS build_tools

RUN yum install -y \
        centos-release-scl-rh \
        epel-release && \
    yum update -y && \
    yum install -y \
        devtoolset-6-toolchain \
        devtoolset-6-libasan-devel \
        devtoolset-6-liblsan-devel \
        tar \
        curl \
        openssl \
        ca-certificates \
        unzip && \
    yum clean all && rm -rf /var/cache/yum

ENV M2_HOME /opt/maven
RUN mkdir -p ${M2_HOME} && \
    curl -fsSL http://apache.osuosl.org/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz | \
        tar -xzC ${M2_HOME} --strip-components=1

RUN mkdir -p /opt/cmake && \
    curl -fsSL https://cmake.org/files/v3.17/cmake-3.17.0-Linux-x86_64.tar.gz | \
    tar -xzC /opt/cmake --strip-components=1

RUN source /opt/rh/devtoolset-6/enable && \
    curl -fsSL https://github.com/google/protobuf/releases/download/v3.8.0/protobuf-cpp-3.8.0.tar.gz \
    | tar xz && \
    cd protobuf-3.8.0 && \
    ./configure --prefix=/opt/protobuf && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf protobuf-3.8.0

ENV NDK_VERSION=r21
RUN mkdir -p /opt && \
    curl -s http://dl.google.com/android/repository/android-ndk-$NDK_VERSION-linux-x86_64.zip -o android-ndk-$NDK_VERSION-linux-x86_64.zip && \
    unzip -q android-ndk-$NDK_VERSION-linux-x86_64.zip -d /opt && \
    rm -f android-ndk-$NDK_VERSION-linux-x86_64.zip

FROM base_image AS base_builder_image

COPY --from=build_tools /opt /opt

RUN yum install -y \
        centos-release-scl-rh \
        epel-release && \
    yum update -y && \
    yum install -y \
        devtoolset-4-toolchain \
        devtoolset-4-libasan-devel \
        devtoolset-4-liblsan-devel \
        devtoolset-6-toolchain \
        devtoolset-6-libasan-devel \
        devtoolset-6-liblsan-devel \
        devtoolset-7-toolchain \
        devtoolset-7-libasan-devel \
        devtoolset-7-liblsan-devel \
        tar \
        wget \
        curl \
        openssl \
        ca-certificates \
        git \
        rpm-build \
        java-1.8.0-openjdk-devel \
        # Required for libnd4j CPU tests (minifier)
        which \
        # Required for Datavec NativeImageLoader
        gtk2-devel && \
    yum clean all && rm -rf /var/cache/yum && \
    ln -s /opt/maven/bin/mvn /usr/bin/mvn




ENV NDK_VERSION=r21

ENV PATH=/opt/sbt/bin:/opt/cmake/bin:/opt/protobuf/bin:${PATH} \
    JAVA_OPTS="-XX:+UnlockExperimentalVMOptions -XX:+UseCGroupMemoryLimitForHeap ${JAVA_OPTS}" \
    ANDROID_NDK_HOME=/opt/android-ndk-$NDK_VERSION

# This environment variable required for buildnativeoperations.sh script execution
ENV ANDROID_NDK=${ANDROID_NDK_HOME}


