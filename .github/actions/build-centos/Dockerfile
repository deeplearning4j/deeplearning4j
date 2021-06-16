FROM centos:7
ADD ./build.sh /build.sh
RUN mv /build.sh /github/workspace/build.sh && chmod +x /github/workspace/build.sh   && /github/workspace/build.sh
