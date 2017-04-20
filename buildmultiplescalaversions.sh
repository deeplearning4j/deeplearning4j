#! /bin/bash
set -eu
./change-scala-versions.sh 2.11 # This should be idempotent : it's the default
mvn "$@"
./change-scala-versions.sh 2.10
mvn "$@"
./change-scala-versions.sh 2.11 # Back to the default
