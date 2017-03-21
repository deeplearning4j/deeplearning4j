#! /bin/bash
set -eu
./change-scala-versions.sh 2.11 # should be idempotent, this is the default
mvn "$@"
./change-spark-versions.sh 2
mvn -Dspark.major.version=2 "$@"
./change-scala-versions.sh 2.10
./change-spark-versions.sh 1
mvn "$@"
./change-scala-versions.sh 2.11 # back to the default
