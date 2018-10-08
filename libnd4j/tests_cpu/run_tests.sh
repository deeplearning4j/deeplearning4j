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


set -exo pipefail

# On Mac, make sure it can find libraries for GCC
export DYLD_LIBRARY_PATH=/usr/local/lib/gcc/8/:/usr/local/lib/gcc/7/:/usr/local/lib/gcc/6/:/usr/local/lib/gcc/5/

# For Windows, add DLLs of MKL-DNN and OpenBLAS to the PATH
if [ -n "$BUILD_PATH" ]; then
    if which cygpath; then
        BUILD_PATH=$(cygpath -p $BUILD_PATH)
    fi
    export PATH="$PATH:$BUILD_PATH"
fi

../blasbuild/cpu/tests_cpu/layers_tests/runtests --gtest_output="xml:../target/surefire-reports/TEST-results.xml"
