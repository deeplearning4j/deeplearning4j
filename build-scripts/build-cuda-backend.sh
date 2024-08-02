#!/bin/bash
cd ..
mvn -Pcuda -Dlibnd4j.chip=cuda clean install -DskipTests -pl :libnd4j,:nd4j-cuda-12.1-preset,:nd4j-cuda-12.1