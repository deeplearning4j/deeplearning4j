ARG BASEIMAGE_VERSION=7
FROM ppc64le/centos:${BASEIMAGE_VERSION} AS base_image

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
        ca-certificates && \
    yum clean all && rm -rf /var/cache/yum

ENV M2_HOME /opt/maven
RUN mkdir -p ${M2_HOME} && \
    curl -fsSL http://apache.osuosl.org/maven/maven-3/3.6.0/binaries/apache-maven-3.6.0-bin.tar.gz | \
        tar -xzC ${M2_HOME} --strip-components=1 && \
    # Workaround for concurrent safe maven local repository
    curl -O http://repo1.maven.org/maven2/io/takari/aether/takari-local-repository/0.11.2/takari-local-repository-0.11.2.jar && \
    mv takari-local-repository-0.11.2.jar ${M2_HOME}/lib/ext && \
    curl -O http://repo1.maven.org/maven2/io/takari/takari-filemanager/0.8.3/takari-filemanager-0.8.3.jar && \
    mv takari-filemanager-0.8.3.jar ${M2_HOME}/lib/ext

RUN mkdir -p /opt/sbt && \
    curl -fsSL https://dl.bintray.com/sbt/native-packages/sbt/0.13.13/sbt-0.13.13.tgz | \
    tar -xzC /opt/sbt --strip-components=1

RUN curl -fsSL https://cmake.org/files/v3.14/cmake-3.14.3.tar.gz \
    | tar xz && \
    cd cmake-3.14.3 && \
    ./configure --prefix=/opt/cmake && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -r cmake-3.14.3

RUN source /opt/rh/devtoolset-6/enable && \
    curl -fsSL https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-cpp-3.5.1.tar.gz \
    | tar xz && \
    cd protobuf-3.5.1 && \
    ./configure --prefix=/opt/protobuf && \
    make -j2 && \
    make install && \
    cd .. && \
    rm -rf protobuf-3.5.1

FROM base_image AS base_builder_image

COPY --from=build_tools /opt /opt

RUN yum install -y \
        centos-release-scl-rh \
        epel-release && \
    yum update -y && \
    yum install -y \
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
        # Required for libnd4j CPU tests
        sudo \
        # Required for libnd4j CPU tests (minifier)
        which \
        # Required for Datavec NativeImageLoader
        gtk2-devel && \
    yum clean all && rm -rf /var/cache/yum && \
    ln -s /opt/maven/bin/mvn /usr/bin/mvn

ENV HOME /home/jenkins

RUN groupadd jenkins -g 1000 && useradd -d ${HOME} -u 1000 -g 1000 -m jenkins

USER jenkins

WORKDIR ${HOME}

RUN mkdir -p .gnupg

# Not working...
RUN echo 'source /opt/rh/devtoolset-7/enable' >> "${HOME}/.bashrc"

ENV PATH=/opt/sbt/bin:/opt/cmake/bin:/opt/protobuf/bin:${PATH} \
    JAVA_OPTS="-XX:+UnlockExperimentalVMOptions -XX:+UseCGroupMemoryLimitForHeap ${JAVA_OPTS}" \
    PROTOBUF_HOME=/opt/protobuf

CMD ["cat"]
