#!/bin/bash
mvn clean package -DskipTests
java -cp target/libnd4j-gen-1.0.0-SNAPSHOT-shaded.jar org.nd4j.descriptor.ParseOpFile "$@"
