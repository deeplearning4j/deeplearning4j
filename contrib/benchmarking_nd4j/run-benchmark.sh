#!/usr/bin/env bash
mvn clean package
 java \
  -jar /home/agibsonccc/Documents/GitHub/deeplearning4j/contrib/benchmarking_nd4j/target/benchmarks.jar \
  -jvm /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests/bin/java \
  org.nd4j.SmallNDArrays

