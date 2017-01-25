#! /bin/bash
set -eu
./change-scala-versions.sh 2.10
mvn "$@"
mvn -Dspark.major.version=2 "$@"
./change-scala-versions.sh 2.11
mvn "$@"
mvn -Dspark.major.version=2 "$@"
./change-scala-versions.sh 2.10
