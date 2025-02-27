#!/bin/bash
cd ..
mvn -Pcuda -Dlibnd4j.compute=86 -Dlibnd4j.chip=cuda clean install -DskipTests -Dlibnd4j.helper=cudnn -pl :libnd4j,:nd4j-cuda-12.1-preset,:nd4j-cuda-12.1 -Dlibnd4j.sanitize=ON -Dlibnd4j.sanitizers=address,undefined,float-divide-by-zero,float-cast-overflow