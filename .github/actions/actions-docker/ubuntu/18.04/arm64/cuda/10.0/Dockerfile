ARG BASEIMAGE_VERSION=r32.2.1
# version: r32.3.1 not working: https://devtalk.nvidia.com/default/topic/1069796/jetson-agx-xavier/cicc-not-found-in-docker-image/
FROM nvcr.io/nvidia/l4t-base:${BASEIMAGE_VERSION} AS base_image

FROM base_image AS build_tools

# Set DEBIAN_FRONTEND to skip any interactive post-install configuration steps
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qqy update && \
    apt-get -y --no-install-recommends install \
        curl \
        ca-certificates \
        build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN curl -fsSL https://cmake.org/files/v3.15/cmake-3.15.5.tar.gz | \
        tar xz && \
        cd cmake-3.15.5 && \
        ./configure --prefix=/opt/cmake && \
        make -j2 && \
        make install && \
        cd .. && \
        rm -r cmake-3.15.5

RUN curl -fsSL https://github.com/google/protobuf/releases/download/v3.8.0/protobuf-cpp-3.8.0.tar.gz \
    | tar xz && \
    cd protobuf-3.8.0 && \
    ./configure --prefix=/opt/protobuf && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -rf protobuf-3.8.0

FROM base_image AS base_builder_image

COPY --from=build_tools /opt /opt
COPY nvidia-deb/libcudnn7*.deb /tmp/
COPY nvidia-deb/cuda-repo-l4t-10-0-local-10.0.326_1.0-1_arm64.deb /tmp

RUN apt-get -qqy update && \
    apt-get -y --no-install-recommends install \
        wget \
        curl \
        ca-certificates \
        ca-certificates-java \
        software-properties-common \
        git \
        build-essential \
        gnupg-agent \
        dirmngr \
        openjdk-8-jdk-headless \
        libopenblas-dev \
        /tmp/libcudnn7*.deb \
        /tmp/cuda-repo-l4t-10-0-local-10.0.326_1.0-1_arm64.deb \
        maven && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    update-java-alternatives -s java-1.8.0-openjdk-arm64 && \
    ln -s /opt/cmake/bin/* /usr/local/bin

# Additional packages for javacpp builds
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F60F4B3D7FA2AF80 && \
    apt-get -qqy update && \
    apt-get -y --no-install-recommends install \
        gcc-arm* \
        gfortran \
        cuda-cudart-dev-10-0 \
        cuda-cublas-dev-10-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
#     apt-get -qqy update && \
#     apt-get -y --no-install-recommends install \
#         gcc-8 \
#         g++-8 && \
#     apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Workaround to fix: CMake Error: The following variables are used in this project, but they are set to NOTFOUND | CUDA_cublas_LIBRARY (ADVANCED) and CUDA_cusolver_LIBRARY (ADVANCED)
RUN cp -a /usr/local/cuda-10.0/lib64/stubs/* /usr/local/cuda-10.0/lib64/ && \
    ln -s /usr/local/cuda-10.0/lib64/libcusolver.so /usr/local/cuda-10.0/lib64/libcusolver.so.10.0

ENV HOME /home/jenkins

RUN groupadd jenkins -g 1000 && useradd -d ${HOME} -u 1000 -g 1000 -m jenkins

USER jenkins

WORKDIR ${HOME}

# RUN mkdir -p .gnupg

ENV PATH=/opt/cmake/bin:/opt/protobuf/bin:${PATH} \
    JAVA_OPTS="-XX:+UnlockExperimentalVMOptions -XX:+UseCGroupMemoryLimitForHeap ${JAVA_OPTS}" \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

CMD ["cat"]
