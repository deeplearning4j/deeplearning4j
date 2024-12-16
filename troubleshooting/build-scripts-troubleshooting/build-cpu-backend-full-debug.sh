#!/bin/bash
# Full debug build for CPU backend with all debug flags enabled

cd ..
mvn -Pcpu clean install -DskipTests \
    -Dlibnd4j.build=debug \
    -Dlibnd4j.calltrace=ON \
    -Dlibnd4j.printmath=ON \
    -Dlibnd4j.printindices=ON \
    -Dlibnd4j.sanitize=ON \
    -Dlibnd4j.sanitizers="address,undefined,float-divide-by-zero,float-cast-overflow" \
    -pl :libnd4j,:nd4j-native-preset,:nd4j-native