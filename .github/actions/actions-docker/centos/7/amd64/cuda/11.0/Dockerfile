ARG BASEIMAGE_VERSION=11.0-cudnn8-devel-centos7

FROM konduit/jenkins-agent:amd64-centos7 AS build_tools

FROM nvidia/cuda:${BASEIMAGE_VERSION}

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
        openblas-devel.x86_64 \
        openblas-static.x86_64 \
        # Required for libnd4j CPU tests (minifier)
        which \
        # Required for Datavec NativeImageLoader
        gtk2-devel \
        # Required for DL4J docs generation
        python-argparse \
        # Required for Datavec-python
        python-pip \
        python36 \
        python36-devel \
        python36-setuptools && \
    yum clean all && rm -rf /var/cache/yum && \
    ln -s /opt/maven/bin/mvn /usr/bin/mvn

# Required for Datavec-python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir Cython --install-option="--no-cython-compile" && \
    easy_install-3.6 pip && \
    pip3 install --upgrade pip && \
    pip3 install --no-cache-dir numpy

# Part of workaround for missing libcuda.so.1 in during dl4j tests
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1




ENV PATH=/opt/sbt/bin:/opt/cmake/bin:/opt/protobuf/bin:${PATH} \
    JAVA_OPTS="-XX:+UnlockExperimentalVMOptions -XX:+UseCGroupMemoryLimitForHeap ${JAVA_OPTS}" \
    PROTOBUF_HOME=/opt/protobuf \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

CMD ["cat"]
