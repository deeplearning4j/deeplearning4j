#!/bin/bash
# CPU backend build with preprocessor output for macro debugging

cd ..
mvn -Pcpu clean install -DskipTests \
    -Dlibnd4j.build=debug \
    -Dlibnd4j.preprocess=ON \
    -pl :libnd4j,:nd4j-native-preset,:nd4j-native