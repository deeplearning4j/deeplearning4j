FROM centos:7
VOLUME /home/runner/
ADD ./build.sh /github/workspace/build.sh
RUN chmod +x /github/workspace/build.sh   && /github/workspace/build.sh
