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

set -eu

# cd to the directory containing this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

install() {
    echo "RUNNING INSTALL"
    if hash nvcc 2>/dev/null; then
    echo "BUILDING CUDA"
        ./buildnativeoperations.sh blas cuda release
        ./buildnativeoperations.sh blas cpu release
    else
        echo "BUILDING CPU"
        ./buildnativeoperations.sh blas cpu release
    fi
}

install
