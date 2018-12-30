---
title: Buidling Deeplearning4j from Source
short_title: Build from Source
description: Instructions to build all DL4J libraries from source.
category: Get Started
weight: 10
---

## Build Locally from Master

**NOTE: MOST USERS SHOULD USE THE RELEASES ON MAVEN CENTRAL AS PER THE QUICK START GUIDE, AND NOT BUILD FROM SOURCE**

*Unless you have a very good reason to build from source (such as developing new features - excluding custom layers, custom activation functions, custom loss functions, etc - all of which can be added without modifying DL4J directly) then you shouldn't build from source. Building from source can be quite complex, with no benefit in a lot of cases. Alternatively, you can use the daily [SNAPSHOT](https://deeplearning4j.org/docs/latest/deeplearning4j-config-snapshots) builds.*

For those developers and engineers who prefer to use the most up-to-date version of Deeplearning4j or fork and build their own version, these instructions will walk you through building and installing Deeplearning4j. The preferred installation destination is to your machine's local maven repository.

Building locally will build the entire Deeplearning4j stack, which is now consolidated inside the Deeplearning4j [monorepo](). The main components of the stack includes:

* [LIBND4J](https://github.com/deeplearning4j/deeplearning4j/tree/master/libnd4j)
* [ND4J](https://github.com/deeplearning4j/deeplearning4j/tree/master/nd4j)
* [Datavec](https://github.com/deeplearning4j/deeplearning4j/tree/master/datavec)
* [Arbiter](https://github.com/deeplearning4j/deeplearning4j/tree/master/arbiter)
* [ND4S](https://github.com/deeplearning4j/deeplearning4j/tree/master/nd4s)
* [Gym Java Client](https://github.com/deeplearning4j/deeplearning4j/tree/master/gym-java-client)
* [RL4J](https://github.com/deeplearning4j/deeplearning4j/tree/master/rl4j)
* [ScalNet](https://github.com/deeplearning4j/deeplearning4j/tree/master/scalnet)
* [PyDL4J](https://github.com/deeplearning4j/deeplearning4j/tree/master/pydl4j)
* [Jumpy](https://github.com/deeplearning4j/deeplearning4j/tree/master/jumpy)
* [PyDatavec](https://github.com/deeplearning4j/deeplearning4j/tree/master/pydatavec)
* [DeepLearning4J](https://github.com/deeplearning4j/deeplearning4j/tree/master/deeplearning4j)

Note that Deeplearning4j is designed to work on most platforms (Windows, OS X, and Linux) and it also includes multiple "flavors" depending on the computing architecture you choose to utilize, such as CUDA for GPUs.

## Prerequisites

Your local machine will require some essential softwares and environment variables set *before* you try to build and install DL4J. Depending on your platform and the version of your operating system, the instructions may vary in getting them to work. This required softwares include:

* Git
* Cmake
* OpenMP 4.5 supported compiler (Recommended)
* GCC >= 4.9
* G++ >= 4.9
* Java >= 1.7
* Maven >= 3.3
* Python Developer Tools
* CUDA SDK >= 8.0 (Optional - For GPU build)

IDE-specific requirements:

* Lombok Plugin for IntelliJ or Eclipse

### Installing Prerequisite Tools

#### Ubuntu

Assuming you are using Ubuntu as your flavor of Linux and you are running as a non-root user, follow these steps to install prerequisite software:

```bash
# Installing JDK, Cmake, git, GCC and G++
sudo apt-get clean -y
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y cmake git openjdk-8-jdk build-essential
echo 'export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-amd64' >> ~/.profile
source ~/.profile

# Installing Maven
MAVEN_VERSION=3.6.0
cd /usr/local/src
sudo wget http://www-eu.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz
sudo tar -xf apache-maven-${MAVEN_VERSION}-bin.tar.gz
sudo rm -f apache-maven-${MAVEN_VERSION}-bin.tar.gz
sudo mv apache-maven-${MAVEN_VERSION}/ apache-maven/
echo 'export PATH=/usr/local/src/apache-maven/bin:${PATH}' >> ~/.profile
source ~/.profile

# Installing Python Developer Tools
sudo apt-get upgrade python-setuptools
sudo apt-get install python-pip python-wheel python-dev
sudo pip install --upgrade pip
sudo pip install --upgrade setuptools
```

#### OS X

Homebrew is the accepted method of installing prerequisite software. Assuming you have Homebrew installed locally, follow these steps to install your necessary tools.

First, before using Homebrew we need to ensure an up-to-date version of Xcode is installed (it is used as a primary compiler):

```bash
xcode-select --install
```

Finally, install prerequisite tools:

```bash
brew update
brew install maven gcc5 python
```

Note: You can *not* use clang. You also can *not* use a new version of gcc. If you have a newer version of gcc, please
switch versions with [this link](https://apple.stackexchange.com/questions/190684/homebrew-how-to-switch-between-gcc-versions-gcc49-and-gcc)

#### Windows

1. Download and Install [Git](https://git-scm.com/download/win)
2. Download and Install `Windows x64` [JDK](https://www.oracle.com/technetwork/java/javase/downloads/index.html) ([Version 8](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html) is recommended)
3. Download and extract the `.zip` file for [Maven](https://maven.apache.org/download.cgi)

Make sure to set the `JAVA_HOME` environment variable to the root JDK installation folder. Also, add the `bin` folder inside the extracted maven folder to the `PATH` environment variable.

`libnd4j` inside the monorepo depends on some Unix utilities for compilation. So in order to compile it you will need to install [Msys2](https://msys2.github.io/).

After you have setup Msys2 by following [their instructions](https://msys2.github.io/), you will have to install some additional development packages. To do that, follow the steps below:

1. Open "C:\msys64\mingw64.ini" and add `MSYS2_PATH_TYPE=inherit`
2. Press `Windows + R` and type `cmd.exe`. Press `Enter` and run the "MSYS2" shell by executing: `c:\msys64\mingw64.exe`

```bash
pacman -Syu # You might have to close the MSYS2 shell and re-run this command, using step 2 above.
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-extra-cmake-modules make pkg-config grep sed gzip tar mingw64/mingw-w64-x86_64-openblas mingw-w64-x86_64-lz4 mingw-w64-x86_64-gdb mingw-w64-x86_64-make mingw-w64-x86_64-ninja
```

This will install the needed dependencies for use in the MSYS2 shell.

You will also need to setup your `PATH` environment variable to include `C:\msys64\mingw64\bin` (or where ever you have decided to install MSYS2). If you have IntelliJ (or another IDE) open, you will have to restart it before this change takes effect for applications started through them. If you don't, you probably will see a "Can't find dependent libraries" error.

#### CentOS

Enter the following in your terminal (or ssh session) as a root user:

```bash
# Installing Cmake, Git
sudo yum clean all && sudo yum update -y && sudo yum install -y git cmake
# Installing JDK
sudo yum install -y java-1.8.0-openjdk-devel
echo 'export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk' >> ~/.bash_profile
source ~/.bash_profile
# Installing Maven
MAVEN_VERSION=3.6.0
cd /usr/local/src
sudo wget http://www-eu.apache.org/dist/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz
sudo tar -xf apache-maven-${MAVEN_VERSION}-bin.tar.gz
sudo rm -f apache-maven-${MAVEN_VERSION}-bin.tar.gz
sudo mv apache-maven-${MAVEN_VERSION}/ apache-maven/
echo 'export PATH=/usr/local/src/apache-maven/bin:${PATH}' >> ~/.bash_profile
source ~/.bash_profile
# Installing GCC and G++
sudo yum install -y centos-release-scl-rh epel-release
sudo yum install -y devtoolset-3-gcc-c++ cmake3
scl enable devtoolset-3 bash
# Installing Python Developer Tools
sudo yum upgrade python-setuptools
sudo yum install python-pip python-wheel python-devel
sudo pip install --upgrade pip
sudo pip install --upgrade setuptools
```

#### CUDA (Optional for GPU build)

CUDA versions starting from `8.0` would work. To install a specific CUDA version in your specific OS, follow the installation instructions on the following website:
https://docs.nvidia.com/cuda/index.html#installation-guides

##### Windows Specific Instructions

For Windows, the CUDA Backend has some additional requirements before it can be built:

1. [CUDA SDK](https://docs.nvidia.com/cuda/index.html#installation-guides) - Any version starting from version `8.0`
2. [Visual Studio](https://visualstudio.microsoft.com/) - Based on the CUDA SDK version, you'll need different versions of Visual Studio.

Visit the following link to confirm which version of Visual Studio is supported with the installed CUDA SDK:
`https://docs.nvidia.com/cuda/archive/`<CUDA_VERSION>`/cuda-toolkit-release-notes/index.html#cuda-compiler-new-features`

Replace `<CUDA_VERSION>` with your installed CUDA_SDK version in the link above. For example, https://docs.nvidia.com/cuda/archive/10.0/cuda-toolkit-release-notes/index.html#cuda-compiler-new-features
                                                                                              
In order to build the CUDA backend you will have to setup some more environment variables first, by calling `vcvars64.bat`.
But first, set the system environment variable `SET_FULL_PATH` to `true`, so all of the variables that `vcvars64.bat` sets up, are passed to the mingw shell.
Additionally, you need to open the `mingw64.ini` in your msys64 installation folder and add the command: `MSYS2_PATH_TYPE=inherit`. After that, follow the steps below:

1. In your visual studio installation folder, press `Ctrl + F` and look for the file `vcvars64.bat`. After it's found, go to its location by right clicking on it and selecting `Open file location` and run it. Open this location inside `cmd.exe` and run the `vcvars64.bat` file. This will set up some environment variables.
2. Inside the console, run `c:\msys64\mingw64.exe`. This will open up the MSYS2 shell with the required environment variables set so you can execute the maven commands to build the DL4J stack for the GPU architecture.

See the section below for the required build commands.

## Build Instructions

First clone the DL4J repository at a valid location and navigate to the `deeplearning4j` folder 

```bash
git clone https://github.com/deeplearning4j/deeplearning4j.git
cd deeplearning4j
```

If you want to change the Scala, Spark or Cuda (for GPU) versions, you can execute one or more of the following commands:
```bash
./change-cuda-versions.sh x.x # Valid versions as of now: (8.0 9.0 9.1 9.2 10.0)
./change-scala-versions.sh 2.xx # Valid versions as of now: (2.10 2.11)
./change-spark-versions.sh x # Valid versions as of now: (1 2)
```

Now, to build for each architecture (CPU or GPU), execute the relevant commands:

### Building for CPU

Let's first set some variables for ease:

```bash
PLATFORM=<YOUR_PLATFORM> # Can be either one of (linux-x86_64, macosx-x86_64, windows-x86_64)
```

The command is: 
```bash
mvn -B -V -U clean install -pl '!deeplearning4j/deeplearning4j-cuda,!nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda,!nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-platform,!jumpy,!pydatavec,!pydl4j' -Dlibnd4j.platform=${PLATFORM} -Djavacpp.platform=${PLATFORM} -P native-snapshots -Dmaven.test.skip=true
```

### Building for GPU

Let's first set some variables for ease:

```bash
PLATFORM=<YOUR_PLATFORM> # Can be either one of (linux-x86_64, macosx-x86_64, windows-x86_64)
CUDA_VERSION=<YOUR_CUDA_SDK_VERSION> # Such as "10.0"
COMPUTE_CAPABILITY=<COMPUTE_CAPABILITY> # Such as "61" for a compute cabability of '6.1'. Can be found here, depending on your GPU type: https://en.wikipedia.org/wiki/CUDA
```

#### Note
 The `COMPUTE_CAPABILITY` variable should be set without any `.` in between. For example, if your GPU has a compute capability of 6.1 then you should set the variable like this:
`COMPUTE_CAPABILITY=61`. Vist this page for more info: https://en.wikipedia.org/wiki/CUDA

The command is: 
```bash
mvn clean install -Dmaven.test.skip -Dlibnd4j.cuda=${CUDA_VERSION} -Dlibnd4j.compute=${COMPUTE_CAPABILITY} -Dlibnd4j.platform=${PLATFORM} -Djavacpp.platform=${PLATFORM}
```

#### IDE Requirements

If you are building Deeplearning4j through an IDE such as IntelliJ, you will need to install certain plugins to ensure your IDE renders code highlighting appropriately. You will need to install a plugin for Lombok:

* IntelliJ Lombok Plugin: https://plugins.jetbrains.com/plugin/6317-lombok-plugin
* Eclipse Lombok Plugin: Follow instructions at https://projectlombok.org/download.html

If you want to work on ScalNet, the Scala API, or on certain modules such as the DL4J UI, you will need to ensure your IDE has Scala support installed and available to you.

#### Testing

Deeplearning4j uses a separate test repository, [dl4j-test-resources](https://github.com/deeplearning4j/dl4j-test-resources), that contains all resources necessary for testing. This is to keep the central DL4J repository lightweight and avoid large blobs in the GIT history. These test resources will automatically be downloaded (~3.5GB in size as of now) when you run the test profiles. 

Tests will run __only__ when `testresources` and a backend profile (such as `test-nd4j-native`) are selected

##### For CPU

Let's first set some variables for ease:

```bash
PLATFORM=<YOUR_PLATFORM> # Can be either one of (linux-x86_64, macosx-x86_64, windows-x86_64)
```

The test command is: 
```bash
mvn clean test -pl '!deeplearning4j/deeplearning4j-cuda,!nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda,!nd4j/nd4j-backends/nd4j-backend-impls/nd4j-cuda-platform,!jumpy,!pydatavec,!pydl4j' -Dlibnd4j.platform=${PLATFORM} -Djavacpp.platform=${PLATFORM} -P testresources,test-nd4j-native
```

##### For GPU

Let's first set some variables for ease:

```bash
PLATFORM=<YOUR_PLATFORM> # Can be either one of (linux-x86_64, macosx-x86_64, windows-x86_64)
CUDA_VERSION=<YOUR_CUDA_SDK_VERSION> # Such as "10.0"
COMPUTE_CAPABILITY=<COMPUTE_CAPABILITY> # Such as "61" for a compute cabability of '6.1'. Can be found here, depending on your GPU type: https://en.wikipedia.org/wiki/CUDA
```

#### Note
 The `COMPUTE_CAPABILITY` variable should be set without any `.` in between. For example, if your GPU has a compute capability of 6.1 then you should set the variable like this:
`COMPUTE_CAPABILITY=61`. Vist this page for more info: https://en.wikipedia.org/wiki/CUDA

The test command is:     
```bash
mvn clean test -Dlibnd4j.cuda=${CUDA_VERSION} -Dlibnd4j.compute=${COMPUTE_CAPABILITY} -Dlibnd4j.platform=${PLATFORM} -Djavacpp.platform=${PLATFORM} -P testresources,test-nd4j-native
```

Running the tests will take a while. To run tests of just a single maven module you can add a module constraint with `-pl deeplearning4j-core` (for details see [here](https://stackoverflow.com/questions/11869762/maven-run-only-single-test-in-multi-module-project))

## Using Local Dependencies

Once you've installed the DL4J stack to your local maven repository, you can now include it in your build tool's dependencies. Follow these pages ([maven](https://deeplearning4j.org/docs/latest/deeplearning4j-config-maven), [SBT, Gradle, & Others](https://deeplearning4j.org/docs/latest/deeplearning4j-config-buildtools)) for your build tool instructions for Deeplearning4j, and appropriately replace versions with the SNAPSHOT version currently on the [master POM](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/pom.xml).

## Support and Assistance

If you encounter issues while building locally, the Deeplearning4j [Early Adopters Channel](https://gitter.im/deeplearning4j/deeplearning4j/earlyadopters) is a channel dedicated to assisting with build issues and other source problems. Please reach out on Gitter for help.