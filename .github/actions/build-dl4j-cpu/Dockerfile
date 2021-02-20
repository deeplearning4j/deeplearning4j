ARG BASEIMAGE_VERSION=18.04

FROM ubuntu:${BASEIMAGE_VERSION} AS base_image

FROM base_image AS build_tools

# Set DEBIAN_FRONTEND to skip any interactive post-install configuration steps
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qqy update && \
    apt-get -y --no-install-recommends install \
        wget \
        curl \
        ca-certificates \
        software-properties-common \
        git \
        build-essential \
        gnupg-agent \
        dirmngr \
        openjdk-8-jdk-headless \
        libssl-dev \
         curl \
           ca-certificates \
        # Required for Datavec NativeImageLoader
        libgtk2.0-dev && \
     apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    update-java-alternatives --set java-1.8.0-openjdk && \
     apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


ENV M2_HOME /opt/maven
RUN mkdir -p ${M2_HOME} && \
    curl -fsSL http://apache.osuosl.org/maven/maven-3/3.6.3/binaries/apache-maven-3.6.3-bin.tar.gz | \
        tar -xzC ${M2_HOME} --strip-components=1

RUN curl -fsSL https://cmake.org/files/v3.19/cmake-3.19.0.tar.gz \
    | tar xz && \
    cd cmake-3.19.0 && \
    ./configure --prefix=/opt/cmake && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -r cmake-3.19.0

RUN curl -fsSL https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.tar.gz \
    | tar xz && \
    cd protobuf-3.5.1 && \
    ./configure --prefix=/opt/protobuf && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -rf protobuf-3.5.1



ENTRYPOINT ["mvn","-Ptestresources","-Ptest-nd4j-native","-Dlibnd4j.chip=cpu","clean","test"]
