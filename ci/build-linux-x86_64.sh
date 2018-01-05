#!/bin/bash
set -evx

if [[ $TRAVIS_PULL_REQUEST == "false" ]]; then
    MAVEN_PHASE="deploy"
else
    MAVEN_PHASE="install"
fi

source change-cuda-versions.sh $CUDA
source change-scala-versions.sh $SCALA
source change-spark-versions.sh $SPARK
mvn clean $MAVEN_PHASE -B -U --settings ./ci/settings.xml -Dmaven.test.skip=true -Dlocal.software.repository=sonatype

