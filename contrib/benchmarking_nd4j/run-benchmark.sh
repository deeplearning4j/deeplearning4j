#!/usr/bin/env bash
#mvn -Djavacpp.platform=linux-x86_64 clean package
 java \
  -jar target/benchmarks.jar \
    -jvm ../../platform-tests/bin/java \
  org.nd4j.Large_NDArray

