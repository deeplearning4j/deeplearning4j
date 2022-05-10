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

function message {
    echo ":::: ${@}"
}

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
BASE_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"


cd ${BASE_DIR}

export VEDNN_ROOT=${BASE_DIR}/vednn_lib
#return

cd ${BASE_DIR}/..
message "install Dl4j"
helper=vednn
extension=avx2
mvn clean  install  -DskipTests  -Dlibnd4j.helper=${helper} -Dlibnd4j.extension=${extension} -Djavacpp.platform.extension=-${helper}-${extension} -Dlibnd4j.classifier=linux-x86_64-${helper}-${extension} -Pcpu

