FROM centos:7
COPY ./build.sh /build.sh
RUN chmod +x /build.sh
ENV JAVA_HOME=/opt/zulu11.58.15-ca-jdk11.0.16-linux_x64
ENTRYPOINT '/build.sh'