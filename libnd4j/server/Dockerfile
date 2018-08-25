#
#  @author raver119@gmail.com
#
FROM ubuntu:bionic

LABEL Description="GraphServer CPU container"  Vendor="SkyMind.ai"  Version="1.0"

# installing basics
RUN apt-get update && apt-get  install -y git maven gcc g++ cmake build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev libprotobuf-dev

# cloning gRPC and DL4J sources
RUN git clone https://github.com/grpc/grpc && git clone https://github.com/deeplearning4j/deeplearning4j && cd grpc && git submodule update --init

# Building gRPC
RUN  cd grpc && make -j 4 && make install

# building libnd4j from sources
RUN  cd deeplearning4j \
                && git checkout r119_twenty_percents \
                && cd libnd4j/server && cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release && make -j 4 && cp GraphServer ../../../

# remove everything
RUN rm -rf grpc deeplearning4j && apt-get purge -y git maven gcc g++ cmake build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev && apt autoremove && apt-get clean -y && rm -rf /var/lib/apt/lists/*

ENV LANG C.UTF-8

EXPOSE 40123/tcp

CMD ["./GraphServer", "-p", "40123"]