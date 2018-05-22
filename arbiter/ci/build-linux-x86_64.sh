#!/bin/bash
set -evx

if [[ $TRAVIS_PULL_REQUEST == "false" ]]; then
    MAVEN_PHASE="deploy"
else
    MAVEN_PHASE="install"
fi

bash change-scala-versions.sh $SCALA
mvn clean $MAVEN_PHASE -B -U --settings ./ci/settings.xml -Dmaven.test.skip=true -Dlocal.software.repository=sonatype

