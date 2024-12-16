#!/bin/bash
# Full debug build for CUDA backend with all debug flags enabled

cd ..
mvn -Pcuda -Dlibnd4j.chip=cuda clean install -DskipTests \
    -Dlibnd4j.build=debug \
    -Dlibnd4j.calltrace=ON \
    -Dlibnd4j.printmath=ON \
    -Dlibnd4j.printindices=ON \
    -Dlibnd4j.sanitize=ON \
    -Dlibnd4j.keepnvcc=ON \
    -Dlibnd4j.sanitizers="address,undefined,float-divide-by-zero,float-cast-overflow" \
    -pl :libnd4j,:nd4j-cuda-12.1-preset,:nd4j-cuda-12.1