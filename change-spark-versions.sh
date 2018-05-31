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

VALID_VERSIONS=( 1 2 )
SPARK_2_VERSION="2\.1\.0"
SPARK_1_VERSION="1\.6\.3"

usage() {
  echo "Usage: $(basename $0) [-h|--help] <spark version to be used>
where :
  -h| --help Display this help text
  valid spark version values : ${VALID_VERSIONS[*]}
" 1>&2
  exit 1
}

if [[ ($# -ne 1) || ( $1 == "--help") ||  $1 == "-h" ]]; then
  usage
fi

TO_VERSION=$1

check_spark_version() {
  for i in ${VALID_VERSIONS[*]}; do [ $i = "$1" ] && return 0; done
  echo "Invalid Spark version: $1. Valid versions: ${VALID_VERSIONS[*]}" 1>&2
  exit 1
}


check_spark_version "$TO_VERSION"

if [ $TO_VERSION = "2" ]; then
  FROM_BINARY="1"
  TO_BINARY="2"
  FROM_VERSION=$SPARK_1_VERSION
  TO_VERSION=$SPARK_2_VERSION
else
  FROM_BINARY="2"
  TO_BINARY="1"
  FROM_VERSION=$SPARK_2_VERSION
  TO_VERSION=$SPARK_1_VERSION
fi

sed_i() {
  sed -e "$1" "$2" > "$2.tmp" && mv "$2.tmp" "$2"
}

export -f sed_i

echo "Updating Spark versions in pom.xml files to Spark $1";

BASEDIR=$(dirname $0)

# <spark.major.version>1</spark.major.version>
find "$BASEDIR" -name 'pom.xml' -not -path '*target*' \
  -exec bash -c "sed_i 's/\(spark.major.version>\)'$FROM_BINARY'<\/spark.major.version>/\1'$TO_BINARY'<\/spark.major.version>/g' {}" \;

# <spark.version>1.6.3</spark.version>
find "$BASEDIR" -name 'pom.xml' -not -path '*target*' \
  -exec bash -c "sed_i 's/\(spark.version>\)'$FROM_VERSION'<\/spark.version>/\1'$TO_VERSION'<\/spark.version>/g' {}" \;

#Spark versions, like <version>xxx_spark_2xxx</version> OR <datavec.spark.version>xxx_spark_2xxx</datavec.spark.version>
find "$BASEDIR" -name 'pom.xml' -not -path '*target*' \
  -exec bash -c "sed_i 's/\(version>.*_spark_\)'$FROM_BINARY'\(.*\)version>/\1'$TO_BINARY'\2version>/g' {}" \;

echo "Done updating Spark versions.";
