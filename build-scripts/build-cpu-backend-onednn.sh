#!/bin/bash
cd ..
mvn -Pcpu clean install -DskipTests -Dlibnd4j.helper=onednn -pl :libnd4j,:nd4j-native-preset,:nd4j-native