#!/bin/bash

git clone https://github.com/deeplearning4j/nd4j.git && cd nd4j/ && mvn clean install -DskipTests -Dmaven.javadoc.skip=true --quiet && cd ../

exit 0
