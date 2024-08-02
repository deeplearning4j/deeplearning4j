#!/bin/bash
cd ..
mvn -Pcuda -Dlibnd4j.compute=86 -Dlibnd4j.chip=cuda clean install -Dlibnd4j.build=debug -Dlibnd4j.calltrace=ON -DskipTests -pl :libnd4j,:nd4j-cuda-12.1-preset,:nd4j-cuda-12.1