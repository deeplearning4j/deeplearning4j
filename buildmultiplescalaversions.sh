#! /bin/bash
set -eu
./change-scala-versions.sh 2.10
mvn "$@"
./change-scala-versions.sh 2.11
mvn -pl "nd4j-serde/nd4j-kryo" "$@"
./change-scala-versions.sh 2.10