#!/usr/bin/env bash
mvn clean package
 java \
  -jar /home/agibsonccc/Documents/GitHub/deeplearning4j/contrib/benchmarking_nd4j/target/benchmarks.jar \
  -jvmArgs   "-agentpath:/home/agibsonccc/YourKit-JavaProfiler-2022.9-b167/YourKit-JavaProfiler-2022.9/bin/linux-x86-64/libyjpagent.so=disablestacktelemetry,exceptions=disable,delay=30000" \
  -jvm /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests/bin/java \
  org.nd4j.SmallNDArrays

