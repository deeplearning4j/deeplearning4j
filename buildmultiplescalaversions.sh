#! /bin/bash
set -eu
./change-scala-versions.sh 2.11 # should be idempotent, this is the default
mvn "$@"
./change-scala-versions.sh 2.10
mvn "$@"
./change-scala-versions.sh 2.11 #back to default
