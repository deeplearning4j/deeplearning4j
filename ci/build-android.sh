#!/bin/bash
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

