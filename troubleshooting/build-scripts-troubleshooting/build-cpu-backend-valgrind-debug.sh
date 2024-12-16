#!/bin/bash
# CPU backend build optimized for Valgrind analysis

cd ..
mvn -Pcpu clean install -DskipTests \
    -Dlibnd4j.build=debug \
    -Dlibnd4j.calltrace=ON \
    -Dlibnd4j.optimization=0 \
    -Dtest.prefix="valgrind --tool=memcheck --track-origins=yes --leak-check=full" \
    -pl :libnd4j,:nd4j-native-preset,:nd4j-native