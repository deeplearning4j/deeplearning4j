ARG BASEIMAGE_VERSION=18.04
FROM ppc64le/ubuntu:${BASEIMAGE_VERSION} AS base_image

FROM base_image AS build_tools

# Set DEBIAN_FRONTEND to skip any interactive post-install configuration steps
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qqy update && \
    apt-get -y --no-install-recommends install \
        curl \
        ca-certificates \
        build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV M2_HOME=/opt/maven

RUN mkdir -p /opt/maven && \
    curl -fsSL http://apache.osuosl.org/maven/maven-3/3.6.2/binaries/apache-maven-3.6.2-bin.tar.gz | \
        tar -xzC /opt/maven --strip-components=1 && \
    # Workaround for concurrent safe maven local repository
    curl -O http://repo1.maven.org/maven2/io/takari/aether/takari-local-repository/0.11.2/takari-local-repository-0.11.2.jar && \
    mv takari-local-repository-0.11.2.jar ${M2_HOME}/lib/ext && \
    curl -O http://repo1.maven.org/maven2/io/takari/takari-filemanager/0.8.3/takari-filemanager-0.8.3.jar && \
    mv takari-filemanager-0.8.3.jar ${M2_HOME}/lib/ext

RUN mkdir -p /opt/sbt && \
    curl -fsSL https://dl.bintray.com/sbt/native-packages/sbt/0.13.13/sbt-0.13.13.tgz | \
    tar -xzC /opt/sbt --strip-components=1

RUN curl -fsSL https://cmake.org/files/v3.15/cmake-3.15.5.tar.gz \
    | tar xz && \
    cd cmake-3.15.5 && \
    ./configure --prefix=/opt/cmake && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -r cmake-3.15.5

RUN curl -fsSL https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.tar.gz \
    | tar xz && \
    cd protobuf-3.5.1 && \
    ./configure --prefix=/opt/protobuf && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -rf protobuf-3.5.1

FROM base_image AS base_builder_image

COPY --from=build_tools /opt /opt

ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-ppc64el \
    M2_HOME=/opt/maven \
    MAVEN_HOME=/opt/maven

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
        # Required for Datavec NativeImageLoader
        libgtk2.0-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    update-java-alternatives --set java-1.8.0-openjdk-ppc64el

RUN mkdir -p .gnupg

ENV PATH=/opt/sbt/bin:/opt/cmake/bin:/opt/protobuf/bin:${MAVEN_HOME}/bin:${PATH} \
    JAVA_OPTS="-XX:+UnlockExperimentalVMOptions -XX:+UseCGroupMemoryLimitForHeap ${JAVA_OPTS}"

CMD ["cat"]

