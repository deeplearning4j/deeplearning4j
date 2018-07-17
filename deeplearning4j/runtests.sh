#!/usr/bin/env bash
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

# Abort on Error
set -e

export PING_SLEEP=30s
export WORKDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export BUILD_OUTPUT=$WORKDIR/build.out

touch $BUILD_OUTPUT

dump_output() {
   echo Tailing the last 500 lines of output:
   tail -500 $BUILD_OUTPUT
}
error_handler() {
  echo ERROR: An error was encountered with the build.
  dump_output
  exit 1
}
# If an error occurs, run our error handler to output a tail of the build
trap 'error_handler' ERR

# Set up a repeating loop to send some output to Travis.

bash -c "while true; do echo \$(date) - building ...; sleep $PING_SLEEP; done" &
PING_LOOP_PID=$!


# My build is using maven, but you could build anything with this, E.g.
# your_build_command_1 >> $BUILD_OUTPUT 2>&1
# your_build_command_2 >> $BUILD_OUTPUT 2>&1
mvn clean test >> $BUILD_OUTPUT 2>&1

# The build finished without returning an error so dump a tail of the output
dump_output

# Repeat for Spark 2
./change-spark-versions.sh 2
mvn clean test -Dspark.major.version=2 >> $BUILD_OUTPUT 2>&1
./change-spark-versions.sh 1

# The build finished without returning an error so dump a tail of the output
dump_output

# nicely terminate the ping output loop
kill $PING_LOOP_PID

