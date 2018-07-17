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

mkdir $HOME/Android/
curl --retry 10 -L https://dl.google.com/android/repository/android-ndk-r16b-linux-x86_64.zip -o $HOME/Android/android-ndk.zip
unzip -qq $HOME/Android/android-ndk.zip -d $HOME/Android/
ln -s $HOME/Android/android-ndk-r16b $HOME/Android/android-ndk
export ANDROID_NDK=$HOME/Android/android-ndk

cd $TRAVIS_BUILD_DIR/../libnd4j/
sed -i /cmake_minimum_required/d CMakeLists.txt
MAKEJ=2 bash buildnativeoperations.sh -platform $OS
cd $TRAVIS_BUILD_DIR/
bash change-scala-versions.sh $SCALA
mvn clean $MAVEN_PHASE -B -U --settings ./ci/settings.xml -Dmaven.javadoc.skip=true -Dmaven.test.skip=true -Dlocal.software.repository=sonatype \
    -Djavacpp.platform=$OS -pl '!nd4j-uberjar,!nd4j-backends/nd4j-backend-impls/nd4j-cuda,!nd4j-backends/nd4j-backend-impls/nd4j-cuda-platform,!nd4j-backends/nd4j-tests'

