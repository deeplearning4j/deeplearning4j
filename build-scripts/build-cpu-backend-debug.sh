#!/bin/bash
cd ..
mvn -Pcpu clean install -Dlibnd4j.calltrace=ON -Dlibnd4j.build=debug -DskipTests -pl :libnd4j,:nd4j-native-preset,:nd4j-native