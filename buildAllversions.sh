#! /bin/bash
set -eu
./change-cuda-versions.sh 8.0 # should be idempotent, this is the default
./buildmultiplescalaversions.sh "$@"
./change-cuda-versions.sh 7.5
./buildmultiplescalaversions.sh "$@"
./change-cuda-versions.sh 8.0 #back to default
