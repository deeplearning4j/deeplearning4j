#!/bin/bash
# CUDA backend build optimized for Compute Sanitizer analysis

cd ..
mvn -Pcuda -Dlibnd4j.chip=cuda clean install -DskipTests \
    -Dlibnd4j.build=debug \
    -Dlibnd4j.calltrace=ON \
    -Dlibnd4j.keepnvcc=ON \
    -Dlibnd4j.optimization=0 \
    -Dtest.runner="compute-sanitizer --tool memcheck" \
    -pl :libnd4j,:nd4j-cuda-12.1-preset,:nd4j-cuda-12.1