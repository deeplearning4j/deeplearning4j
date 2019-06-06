#! /bin/bash
################################################################################
# Copyright (c) 2015-2018 Skymind, Inc.
#
# This program and the accompanying materials are made available under the
# terms of the Apache License, Version 2.0 which is available at
# https://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
#
# SPDX-License-Identifier: Apache-2.0
################################################################################

BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

function echoError() {
    (>&2 echo "$1")
}

function scalaError() {
    echoError "Changing Scala major version to 2.10 in the build did not change the state of your working copy, is Scala 2.11 still the default ?"
    exit 2
}

function whatchanged() {
    cd "$BASEDIR"
    for i in $(git status -s --porcelain -- $(find ./ -mindepth 2 -name pom.xml)|awk '{print $2}'); do
        echo "$(dirname $i)"
        cd "$BASEDIR"
    done
}

set -eu
./change-scala-versions.sh 2.11 # should be idempotent, this is the default
mvn "$@"
./change-scala-versions.sh 2.10
if [ -z "$(whatchanged)" ]; then
    scalaError;
else
    if [[ "${@#-pl}" = "$@" ]]; then
        mvn -Dmaven.clean.skip=true -pl $(whatchanged| tr '\n' ',') -amd "$@"
    else
        # the arguments already tweak the project list ! don't tweak them more
        # as this can lead to conflicts (excluding a project that's not part of
        # the reactor)
        mvn "$@"
    fi
fi
./change-scala-versions.sh 2.11 # back to the default
