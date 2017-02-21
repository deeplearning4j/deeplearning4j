#! /bin/bash
set -eu
mvn "$@"
./change-spark-versions.sh 2
mvn -Dspark.major.version=2 "$@"
./change-scala-versions.sh 2.10
./change-spark-versions.sh 1
mvn "$@"
./change-scala-versions.sh 2.11
