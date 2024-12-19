#!/usr/bin/env bash
mvn clean package
 java \
  -jar target/benchmarks.jar \
    -jvm ../platform-tests/bin/java \
  org.nd4j.SmallNDArrays

