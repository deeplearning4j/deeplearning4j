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

brew update
brew upgrade maven || true
brew install gcc || true
brew link --overwrite gcc

/usr/local/bin/gcc-? --version
mvn -version

if [[ $CUDA == "8.0" ]]; then
    curl --retry 10 -L https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_mac-dmg -o $HOME/cuda.dmg
else
    curl --retry 10 -L https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_mac-dmg -o $HOME/cuda.dmg
fi
hdiutil mount $HOME/cuda.dmg
sleep 5
sudo /Volumes/CUDAMacOSXInstaller/CUDAMacOSXInstaller.app/Contents/MacOS/CUDAMacOSXInstaller --accept-eula --no-window

cd $TRAVIS_BUILD_DIR/../libnd4j/
sed -i="" /cmake_minimum_required/d CMakeLists.txt
MAKEJ=2 bash buildnativeoperations.sh -c cpu -e $EXT
MAKEJ=1 bash buildnativeoperations.sh -c cuda -v $CUDA -cc 30
cd $TRAVIS_BUILD_DIR/
source change-cuda-versions.sh $CUDA
source change-scala-versions.sh $SCALA
mvn clean $MAVEN_PHASE -B -U --settings ./ci/settings.xml -Dmaven.javadoc.skip=true -Dmaven.test.skip=true -Dlocal.software.repository=sonatype -Djavacpp.extension=$EXT

