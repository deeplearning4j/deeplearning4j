#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This shell script is adapted from Apache Flink (in turn, adapted from Apache Spark) some modifications.

set -e

VALID_VERSIONS=( 7.5 8.0 )
CUDA_75_VERSION="7\.5"
CUDA_80_VERSION="8\.0"

usage() {
  echo "Usage: $(basename $0) [-h|--help] <cuda version to be used>
where :
  -h| --help Display this help text
  valid cuda version values : ${VALID_VERSIONS[*]}
" 1>&2
  exit 1
}

if [[ ($# -ne 1) || ( $1 == "--help") ||  $1 == "-h" ]]; then
  usage
fi

TO_VERSION=$1

check_cuda_version() {
  for i in ${VALID_VERSIONS[*]}; do [ $i = "$1" ] && return 0; done
  echo "Invalid CUDA version: $1. Valid versions: ${VALID_VERSIONS[*]}" 1>&2
  exit 1
}


check_cuda_version "$TO_VERSION"

if [ $TO_VERSION = "8.0" ]; then
  FROM_BINARY="-7\.5"
  TO_BINARY="-8\.0"
  FROM_VERSION=$CUDA_75_VERSION
  TO_VERSION=$CUDA_80_VERSION
else
  FROM_BINARY="-8\.0"
  TO_BINARY="-7\.5"
  FROM_VERSION=$CUDA_80_VERSION
  TO_VERSION=$CUDA_75_VERSION
fi

sed_i() {
  sed -e "$1" "$2" > "$2.tmp" && mv "$2.tmp" "$2"
}

export -f sed_i

echo "Updating CUDA versions in pom.xml files to CUDA $1";

BASEDIR=$(dirname $0)

#Artifact ids, ending with "-7.5" or "-8.0". nd4j-cuda, deeplearning4j-cuda, etc.
find "$BASEDIR" -name 'pom.xml' -not -path '*target*' \
  -exec bash -c "sed_i 's/\(artifactId>.*\)'$FROM_BINARY'<\/artifactId>/\1'$TO_BINARY'<\/artifactId>/g' {}" \;

#Artifact ids, ending with "-7.5-platform" or "-8.0-platform". nd4j-cuda-platform, etc.
find "$BASEDIR" -name 'pom.xml' -not -path '*target*' \
  -exec bash -c "sed_i 's/\(artifactId>.*\)'$FROM_BINARY'-platform<\/artifactId>/\1'$TO_BINARY'-platform<\/artifactId>/g' {}" \;

#CUDA versions, like <cuda.version>7.5</cuda.version>
find "$BASEDIR" -name 'pom.xml' -not -path '*target*' \
  -exec bash -c "sed_i 's/\(cuda.version>\)'$FROM_VERSION'<\/cuda.version>/\1'$TO_VERSION'<\/cuda.version>/g' {}" \;

echo "Done updating CUDA versions.";
