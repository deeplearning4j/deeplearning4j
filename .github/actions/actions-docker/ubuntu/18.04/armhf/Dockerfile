ARG BASEIMAGE_VERSION=16.04
FROM amd64/ubuntu:${BASEIMAGE_VERSION} AS base_image

FROM base_image AS build_tools

# Set DEBIAN_FRONTEND to skip any interactive post-install configuration steps
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qqy update && \
    apt-get -y --no-install-recommends install \
        curl \
        git \
        ca-certificates \
        build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir -p /opt/maven && \
    curl -fsSL http://apache.osuosl.org/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz | \
    tar -xzC /opt/maven --strip-components=1

RUN mkdir -p /opt/cmake && \
    curl -fsSL https://cmake.org/files/v3.15/cmake-3.15.5-Linux-x86_64.tar.gz | \
    tar -xzC /opt/cmake --strip-components=1

RUN curl -fsSL https://github.com/google/protobuf/releases/download/v3.8.0/protobuf-cpp-3.8.0.tar.gz | \
    tar xz && \
    cd protobuf-3.8.0 && \
    ./configure --prefix=/opt/protobuf && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf protobuf-3.8.0

WORKDIR /tmp

ENV RPI_HOME=/opt/raspberrypi

RUN mkdir -p $RPI_HOME

RUN cd $RPI_HOME && \
    git clone --depth=1 git://github.com/raspberrypi/tools.git

FROM base_image AS base_builder_image

COPY --from=build_tools /opt /opt

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
    M2_HOME=/opt/maven \
    MAVEN_HOME=/opt/maven

RUN apt-get -qqy update && \
    apt-get -y --no-install-recommends install \
        wget \
        curl \
        ca-certificates \
        software-properties-common \
        git \
        gnupg-agent \
        dirmngr \
        openjdk-8-jdk-headless \
        # Required for Raspberry PI cross compilation tools
        lib32stdc++6 \
        lib32z1 \
        lib32ncurses5 \
        # Required for Datavec NativeImageLoader
        libgtk2.0-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    update-java-alternatives --set java-1.8.0-openjdk-amd64



ENV RPI_HOME=/opt/raspberrypi
ENV PATH=$RPI_HOME/tools/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian/bin:/opt/cmake/bin:/opt/protobuf/bin:${MAVEN_HOME}/bin:${PATH} \
    JAVA_OPTS="-XX:+UnlockExperimentalVMOptions -XX:+UseCGroupMemoryLimitForHeap ${JAVA_OPTS}" \
    LD_LIBRARY_PATH=$RPI_HOME/tools/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian/lib:$LD_LIBRARY_PATH \
    RPI_BIN=$RPI_HOME/tools/arm-bcm2708/gcc-linaro-arm-linux-gnueabihf-raspbian/bin/arm-linux-gnueabihf

CMD ["cat"]