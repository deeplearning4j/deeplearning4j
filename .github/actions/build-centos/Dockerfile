FROM centos:7
COPY ./build.sh /build.sh
RUN chmod +x /build.sh
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.8.10-1.el7.x86_64
ENTRYPOINT '/build.sh'