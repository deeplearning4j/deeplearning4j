#!/bin/bash
export PATH=/usr/local/cuda-12.9/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.9
rm -rf libnd4j/blasbuild
cd /home/agibsonccc/Documents/GitHub/deeplearning4j
mvn install -pl :libnd4j -DskipTests -Dlibnd4j.compute="89" -Pcuda -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=1
