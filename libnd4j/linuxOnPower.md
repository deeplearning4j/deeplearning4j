# Build DL4J for Linux on Power

This document explains how to build DL4J for Linux on Power

## Check pre-requirements

The following tools are necessary to build DL4J for Linux on Power

1. gcc version 4.9.X (older or newer version may cause build errors.)
   Check the version by "gcc --version", and install it from https://gcc.gnu.org/ if necessary.
   Look at Apppendix of this document to install gcc 4.9.

2. OpenJDK java 1.8
   Check the version is OpenJDK and 1.8 by "java -version".

3. Apache Maven 3.3 or later
   Check the version by "mvn -v", and install it from https://maven.apache.org/ if necessary.

4. CUDA7.5 (CUDA7.0 is not supported)
   Check the version by "nvcc -V"

5. cmake version 3.5.0 or later

## Set Environment Variables

Edit the CUDA, JAVA_HOME, CC, CXX environment variables according to your system

```
export CUDA=/usr/local/cuda-7.5				# CUDA Directory
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-ppc64el	# JAVA Directory
export CC=/path/to/gcc					# gcc command for build
export CXX=/path/to/g++					# g++ command for build
export MAVEN_OPTS='-Xmx4096M -Dos.arch=ppc64le'
export _JAVA_OPTIONS=-Dos.arch=ppc64le
export LIBND4J_HOME=`/bin/pwd`/libnd4j
export CUDA_VISIBLE_DEVICES=0
```

## Clone and build DL4J, and execute a sample program

1. Clone necessary modules from github as follows.
```
git clone https://github.com/deeplearning4j/Canova
git clone https://github.com/deeplearning4j/nd4j.git
git clone https://github.com/deeplearning4j/libnd4j.git
git clone https://github.com/deeplearning4j/dl4j-0.4-examples.git
git clone https://github.com/deeplearning4j/deeplearning4j.git
git clone https://github.com/bytedeco/javacpp.git
```

2. Modify the setting file to enable GPU. (Skip this step if you do not use GPU)
Modify the following line in dl4j-0.4-examples/pom.xml as follows.
```
Before:  <nd4j.backend>nd4j-native</nd4j.backend>
After:    <nd4j.backend>nd4j-cuda-7.5</nd4j.backend>
```

3. Build modules as follows. (Make sure you follow the instructions in order.)
```
(cd javacpp/; mvn clean install -DskipTests)
(cd libnd4j; ./buildnativeoperations.sh)
(cd libnd4j; ./buildnativeoperations.sh -c cuda)
(cd nd4j; mvn -e clean install -DskipTests -DskipTests -Djavacpp.platform.dependency=false -Dmaven.javadoc.skip=true)
(cd Canova/; mvn clean install -DskipTests -Djavacpp.platform.dependency=false)
(cd deeplearning4j; mvn clean package -DskipTests -Djavacpp.platform.dependency=false)
(cd dl4j-0.4-examples; mvn clean package -DskipTests)
```

4. Test the module by running the LenetMnist example
```
(cd dl4j-0.4-examples; java -cp target/deeplearning4j-examples-0.4-rc0-SNAPSHOT-bin.jar org.deeplearning4j.examples.convolution.LenetMnistExample)
```

## Appendix

How to build gcc 4.9.3 on Linux on Power

```
$ tar xvfz gcc-4.9.3.tar.gz
$ cd gcc-4.9.3
$ mkdir -p build
$ (cd build; ../configure --enable-languages=c,c++ --prefix=<install path> --disable-bootstrap --disable-multilib)
$ (cd build; make)
$ (cd build; make install)
# You need to add <install path>/lib64 in LD_LIBRRY_PATH
# BLAS need to be specified in LD_LIBRARY_PATH to run CPU(native) version
```
