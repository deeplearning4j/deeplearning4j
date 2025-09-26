#!/usr/bin/env bash

#
# /* ******************************************************************************
#  *
#  *
#  * This program and the accompanying materials are made available under the
#  * terms of the Apache License, Version 2.0 which is available at
#  * https://www.apache.org/licenses/LICENSE-2.0.
#  *
#  *  See the NOTICE file distributed with this work for additional
#  *  information regarding copyright ownership.
#  * Unless required by applicable law or agreed to in writing, software
#  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  * License for the specific language governing permissions and limitations
#  * under the License.
#  *
#  * SPDX-License-Identifier: Apache-2.0
#  ******************************************************************************/
#





set -e

VALID_VERSIONS=( 9.2 10.0 10.1 10.2 11.0 11.1 11.2 11.4 11.6 11.8 12.1 12.3 12.6)

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

VERSION=$1

check_cuda_version() {
  for i in ${VALID_VERSIONS[*]}; do [ $i = "$1" ] && return 0; done
  echo "Invalid CUDA version: $1. Valid versions: ${VALID_VERSIONS[*]}" 1>&2
  exit 1
}


check_cuda_version "$VERSION"

case $VERSION in
 12.6)
    VERSION2="9.5"
    VERSION3="1.5.11"
    ;;
 12.3)
    VERSION2="8.9"
    VERSION3="1.5.10"
    ;;
 12.1)
    VERSION2="8.9"
    VERSION3="1.5.9"
    ;;
 11.8)
    VERSION2="8.6"
    VERSION3="1.5.8"
    ;;
 11.6)
    VERSION2="8.3"
    VERSION3="1.5.7"
    ;;
  11.4)
    VERSION2="8.2"
    VERSION3="1.5.6"
    ;;
  11.2)
    VERSION2="8.1"
    VERSION3="1.5.5"
    ;;
  11.1)
    VERSION2="8.0"
    VERSION3="1.5.5"
    ;;
  11.0)
    VERSION2="8.0"
    VERSION3="1.5.4"
    ;;
  10.2)
    VERSION2="8.2"
    VERSION3="1.5.6"
    ;;
  10.1)
    VERSION2="7.6"
    VERSION3="1.5.2"
    ;;
  10.0)
    VERSION2="7.4"
    VERSION3="1.5"
    ;;
  9.2)
    VERSION2="7.2"
    VERSION3="1.5"
    ;;
esac

sed_i() {
  if test -f "$2" && test -f "$1"; then
     sed -i "" -e "$1" "$2" > "$2.tmp" && mv "$2.tmp" "$2"
  fi
}

export -f sed_i

echo "Updating CUDA versions in pom.xml files to CUDA $1";

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
BASEDIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

cd "${BASEDIR}/"contrib/version-updater
mvn clean compile
mvn exec:java -Dexec.args="--root-dir=${BASEDIR} --cuda-version=${VERSION} --cudnn-version=${VERSION2} --javacpp-version=${VERSION3} --update-type=cuda"

echo "Done updating CUDA versions.";
