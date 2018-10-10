# Monorepo of Deeplearning4j

Welcome to the new monorepo of Deeplearning4j that contains the source code for all the following projects, in addition to the original repository of Deeplearning4j moved to [deeplearning4j](deeplearning4j):

 * https://github.com/deeplearning4j/libnd4j
 * https://github.com/deeplearning4j/nd4j
 * https://github.com/deeplearning4j/datavec
 * https://github.com/deeplearning4j/arbiter
 * https://github.com/deeplearning4j/nd4s
 * https://github.com/deeplearning4j/gym-java-client
 * https://github.com/deeplearning4j/rl4j
 * https://github.com/deeplearning4j/scalnet
 * https://github.com/deeplearning4j/jumpy

To build everything, we can use commands like
```
./change-cuda-versions.sh x.x
./change-scala-versions.sh 2.xx
./change-spark-versions.sh x
mvn clean install -Dmaven.test.skip -Dlibnd4j.cuda=x.x -Dlibnd4j.compute=xx
```
or
```
mvn clean install -Ptestresources -Ptest-native -Dlibnd4j.cuda=x.x -Dlibnd4j.compute=xx
```

# Want some examples?
We have separate repository with various examples available: https://github.com/deeplearning4j/dl4j-examples

In the examples repo, you'll also find a tutorial series in Zeppelin: https://github.com/deeplearning4j/dl4j-examples/tree/master/tutorials



# Detailed Instructions to build DL4J
## Environment Setup (This instructions has been tested on Ubuntu 16.04)

## Get and install Ubuntu

You can get ubuntu from here: [[http://releases.ubuntu.com/]]

## Install Nvidia driver (if you have Nvidia cuda-capable graphic card)

The easy option is this:
```
sudo add-apt-repoistory ppa:xorg-edgers/ppa
sudo apt-get update
sudo apt-get install nvidia-_YourPrefferedVersion_
```

## Install Java Development Kit (8)

You can install oracle or openjdk. Oracle has been tested.

## Install Cuda Sdk

You can obtain its docs here: [[http://docs.nvidia.com/cuda/]]

## Install Nexus Repository Manager (preferred to speedup your build phase)

You can view its documentation here. [[https://help.sonatype.com/repomanager3/installation]]

## Now your environment is good and ready. Let's head to the Building Steps.

## Requirements:

 * git
 * cmake (3.6 or higher)
 * OpenMP
 * gcc (4.9 or higher)
 * maven (3.3 or higher)

Check the cmake and maven version. They have to be at lease the mentioned version.
If default cmake in Ubuntu was not higher than 3.6, you have to install it manually:

```
sudo apt-get remove cmake
sudo apt-get purge --auto-remove cmake
```

Then download cmake from: http://www.cmake.org/download
Extract it by `tar xzf nameOfThefile.tar.gz` and cd into it. like:

```
cd cmake-3.13.0-rc1/
./bootstrap
make 
sudo make install
```

check the version again.

## Install other Requirements

```
sudo apt-get purge maven maven2 maven3
sudo add-apt-repository ppa:natecarlson/maven3
sudo apt-get update
sudo apt-get install maven build-essential libgomp1 git
```

## Install Prerequisite Architecture

### OpenBLAS

```
sudo apt-get install libopenblas-dev
```

You will also need to ensure that /opt/OpenBLAS/lib (or any other home directory for OpenBLAS)
is on your PATH. In order to get OpenBLAS to work with Apache Spark, 
you will also need to make sure that libraries `liblapack.so.3` and `libblas.so.3` are present in /usr/lib/openblas-base.
If they don't exist you can do the following:

```
sudo cp libopenblas.so liblapack.so.3
sudo cp libopenblas.so libblas.so.3
```

### ATLAS

```
sudo apt-get install libatlas-base-dev libatlas-dev
```

# Installing the DL4J Stack

## Get the source code:

```
git clone https://github.com/deeplearning4j/deeplearning4j.git
```

It will take some time according to your network speed.
Then you have to checkout to a stable version of the code.
And then create a simple branch.

```
git checkout latest_release
git checkout -b newBranch
```
Before running the DL4J stack build script, you must ensure 
certain environment variables are defined before running your build.

You will need to know the exact path of the directory where you are running
the DL4J build script (you are encouraged to use a clean empty directory).
Otherwise, your build will fail. Once you determine this path, add /libnd4j
to the end of that path and export it to your local environment. This will look like:

```
export LIBND4J_HOME="/path/to/deeplearning4j/libnd4j"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIBND4J_HOME/include
```
You can add those lines at the end of your `bashrc` file.


## Build Commands

When you are in the base folder run these commands. assume that:

 * cuda version is 8.0
 * scala version is 2.11
 * spark version is 2

```
./change-cuda-versions.sh 8.0
./change-scala-versions.sh 2.11
./change-spark-versions.sh 2
```

if you don't want to use Cuda, build with this command:

```
mvn clean install -Dmaven.test.skip -DskipTests -Dmaven.javadoc.skip=true -pl '!./nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda,!./nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-platform,!./nd4j/nd4j-backends/nd4j-tests,!./deeplearning4j/deeplearning4j-cuda/'
```

And run this command if you have intend to use Cuda:
replace `libnd4j.cuda=x.x` with your cuda version, like `-Dlibnd4j.cuda=8.0`
replace `libnd4j.compute=xx` with your gpu capabality, like `-Dlibnd4j.compute=30` for GTX 680.
Checkout your device compute capability here: https://en.wikipedia.org/wiki/CUDA

```
mvn clean install -Dlibnd4j.cuda=x.x -Dlibnd4j.compute=xx -Dmaven.test.skip -DskipTests -Dmaven.javadoc.skip=true -pl '!./nd4j/nd4j-backends/nd4j-tests'
```

Sometimes I had to use -U for maven to force update the packages through Nexus. 
And I had to add this maven repository(proxy) to the Nexus: [[http://maven.restlet.org]] 
You can search to find how to do this. It's really simple forward.

## You can check DL4J tutorials and docs pages for more.

here: [[https://deeplearning4j.org/docs/latest/]] and here: [[https://deeplearning4j.org/tutorials/]]
