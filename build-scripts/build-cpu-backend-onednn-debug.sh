#!/bin/bash
cd ..
mvn -Pcpu clean install -DskipTests -Dlibnd4j.calltrace=ON -Dlibnd4j.helper=onednn  -Dlibnd4j.build=debug -pl :libnd4j,:nd4j-native-preset,:nd4j-native