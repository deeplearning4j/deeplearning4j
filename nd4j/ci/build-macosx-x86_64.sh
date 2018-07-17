#!/bin/bash
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

set -evx

while true; do echo .; sleep 60; done &

if [[ $TRAVIS_PULL_REQUEST == "false" ]]; then
    BRANCH=$TRAVIS_BRANCH
    MAVEN_PHASE="deploy"
else
    BRANCH=$TRAVIS_PULL_REQUEST_BRANCH
    MAVEN_PHASE="install"
fi

if ! git -C $TRAVIS_BUILD_DIR/.. clone https://github.com/deeplearning4j/libnd4j/ --depth=50 --branch=$BRANCH; then
     git -C $TRAVIS_BUILD_DIR/.. clone https://github.com/deeplearning4j/libnd4j/ --depth=50
fi

brew update
brew upgrade maven || true
brew install gcc || true
brew link --overwrite gcc

/usr/local/bin/gcc-? --version
mvn -version

if [[ "${CUDA:-}" == "8.0" ]]; then
    CUDA_URL=https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_mac-dmg
elif [[ "${CUDA:-}" == "9.0" ]]; then
    CUDA_URL=https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_mac-dmg
elif [[ "${CUDA:-}" == "9.1" ]]; then
    CUDA_URL=https://developer.nvidia.com/compute/cuda/9.1/Prod/local_installers/cuda_9.1.85_mac
fi
if [[ -n ${CUDA_URL:-} ]]; then
    curl --retry 10 -L -o $HOME/cuda.dmg $CUDA_URL
    hdiutil mount $HOME/cuda.dmg
    sleep 5
    sudo /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS/CUDAMacOSXInstaller --accept-eula --no-window
fi

cd $TRAVIS_BUILD_DIR/../libnd4j/
sed -i="" /cmake_minimum_required/d CMakeLists.txt
if [[ -n "${CUDA:-}" ]]; then
    MAKEJ=1 bash buildnativeoperations.sh -c cuda -v $CUDA -cc 30
    cd $TRAVIS_BUILD_DIR/
    bash change-cuda-versions.sh $CUDA
    EXTRA_OPTIONS='-pl !nd4j-uberjar,!nd4j-backends/nd4j-backend-impls/nd4j-native,!nd4j-backends/nd4j-backend-impls/nd4j-native-platform,!nd4j-backends/nd4j-tests'
else
    MAKEJ=2 bash buildnativeoperations.sh -c cpu -e ${EXT:-}
    cd $TRAVIS_BUILD_DIR/
    EXTRA_OPTIONS='-pl !nd4j-uberjar,!nd4j-backends/nd4j-backend-impls/nd4j-cuda,!nd4j-backends/nd4j-backend-impls/nd4j-cuda-platform,!nd4j-backends/nd4j-tests'
fi
bash change-scala-versions.sh $SCALA
mvn clean $MAVEN_PHASE -B -U --settings ./ci/settings.xml -Dmaven.javadoc.skip=true -Dmaven.test.skip=true -Dlocal.software.repository=sonatype \
    -Djavacpp.extension=${EXT:-} $EXTRA_OPTIONS

