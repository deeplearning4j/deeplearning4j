---
title: Building the DL4J Stack Locally
layout: default
---

# Build Locally from Master

For those developers and engineers who prefer to use the most up-to-date version of Deeplearning4j or fork and build their own version, these instructions will walk you through building and installing Deeplearning4j. The preferred installation destination is to your machine's local maven repository. If you are not using the master branch, you can modify these steps as needed (i.e.: switching GIT branches and modifying the `build-dl4j-stack.sh` script).

Building locally requires that you build the entire Deeplearning4j stack which includes:

- [libnd4j](https://github.com/deeplearning4j/libnd4j)
- [nd4j](https://github.com/deeplearning4j/nd4j)
- [deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)

Note that Deeplearning4j is designed to work on most platforms (Windows, OS X, and Linux) and is also includes multiple "flavors" depending on the computing architecture you choose to utilize. This includes CPU (OpenBLAS, MKL, ATLAS) and GPU (CUDA). The DL4J stack also supports x86 and PowerPC architectures.

## Prerequisites

Your local machine will require some essential software and environment variables set *before* you try to build and install the DL4J stack. Depending on your platform and the version of your operating system, the instructions may vary in getting them to work. This software includes:

- git
- cmake with OpenMP
- gcc (3.2 or higher)
- maven (3 or higher)

Architecture-specific software includes:

CPU options:

- Intel MKL
- OpenBLAS
- ATLAS
- ND4J-Java (Android)

GPU options:

- Jcublas/CUDA
- JOCL (coming soon)

### Installing Prerequisite Tools

#### Linux

**Ubuntu**
Assuming you are using Ubuntu as your flavor of Linux and you are running as a non-root user, follow these steps to install prerequisite software:

```
sudo apt-get purge maven maven2 maven3
sudo add-apt-repository ppa:natecarlson/maven3
sudo apt-get update
sudo apt-get install maven build-essentials cmake libgomp1

```

#### OS X

Homebrew is the accepted method of installing prerequisite software. Assuming you have Homebrew installed locally, follow these steps to install your necessary tools.

First, before using Homebrew we need to ensure an up-to-date version of Xcode is installed (it is used as a primary compiler):

```
xcode-select --install
```

Finally, install prerequisite tools:

```
brew update
brew install maven clang-omp
```

#### Windows

Windows users may need to install [Visual Studio Community 2013 or later](https://www.visualstudio.com/en-us/products/visual-studio-community-vs.aspx), which is free. You will need to add its path to your PATH environment variable manually. The path will look something like this: `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin`

Type `cl` into your CMD. You may get a message informing you that certain `.dll` files are missing. Make sure that your VS/IDE folder is in the path (see above). If your CMD returns usage info for `cl`, then it's in the right place. 

If you use Visual Studio: 

1. Set up `PATH` environment variable to point to `\bin\` (for `cl.exe` etc)
1. Also try running `vcvars32.bat` (also in bin) to set up environment before doing `mvn clean install` on ND4J (it may save you from copying headers around)
1. `vcvars32` may be temporary, so you might need to run it every time you want to do ND4J `mvn install`.
1. After installing Visual Studio 2015 and setting the PATH variable, you need to run the `vcvars32.bat` to set up the environment variables (INCLUDE, LIB, LIBPATH) properly so that you don't have to copy header files. But if you run the bat file from Explorer, since the settings are temporary, they're not properly set. So run `vcvars32.bat` from the same CMD window as your `mvn install`, and all the environment variables will be set correctly.
1. Here is how they should be set: 

    INCLUDE = C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\include
    LIB = "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\lib"
    //so you can link to .lib files^^
1. In Visual Studio, you also have to click on C++. It is no longer set by default. 
(*In addition, the include path for [Java CPP](https://github.com/bytedeco/javacpp) doesn't always work on Windows. One workaround is to take the the header files from the include directory of Visual Studio, and put them in the include directory of the Java Run-Time Environment (JRE), where Java is installed. This will affect files such as `standardio.h`.*)
* For a walkthrough of installing our examples with Git, IntelliJ and Maven, please see our [Quickstart page](http://deeplearning4j.org/quickstart.html#walk)

### Installing Prerequisite Architectures

Once you have installed the prerequisite tools, you can now install the required architectures for your platform.

#### CPU

##### Intel MKL

Of all the existing architectures available for CPU, Intel MKL is currently the fastest. However, it requires some "overhead" before you actually install it.

1. Apply for a license at [Intel's site](https://software.intel.com/en-us/intel-mkl)
2. After a few steps through Intel, you will receive a download link
3. Download and install Intel MKL using [the setup guide](https://software.intel.com/sites/default/files/managed/94/bf/Install_Guide_0.pdf)

##### OpenBLAS

###### Linux

**Ubuntu**
Assuming you are using Ubuntu, you can install OpenBLAS via:

```
sudo apt-get install libopenblas-dev
```

You will also need to ensure that `/opt/OpenBLAS/lib` (or any other home directory for OpenBLAS) is on your `PATH`. In order to get OpenBLAS to work with Apache Spark, you will also need to do the following:

```
sudo cp libopenblas.so liblapack.so.3
sudo cp libopenblas.so libblas.so.3
```

**CentOS**
Enter the following in your terminal (or ssh session) as a root user:

    yum groupinstall 'Development Tools'

After that, you should see a lot of activity and installs on the terminal. To verify that you have, for example, *gcc*, enter this line:

    gcc --version

For more complete instructions, [go here](http://www.cyberciti.biz/faq/centos-linux-install-gcc-c-c-compiler/).

###### OS X

You can install OpenBLAS on OS X with Homebrew Science:

```
brew install homebrew/science/openblas
```

###### Windows

[This page](http://avulanov.blogspot.cz/2014/09/howto-to-run-netlib-javabreeze-in.html) describes how to obtain dll for the Windows 64 platform:
1. Download dll libraries and place them in the Java bin folder (e.g. `C:\prg\Java\jdk1.7.0_45\bin`).
2. Library `netlib-native_system-win-x86_64.dll` depends on: 
`libgcc_s_seh-1.dll
libgfortran-3.dll
libquadmath-0.dll
libwinpthread-1.dll
libblas3.dll
liblapack3.dll`
3. (`liblapack3.dll` and `libblas3.dll` are just renamed copies of `libopeblas.dll`)
4. You can download compiled libs from [here](http://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win64/Automated%20Builds/), [here](http://www.openblas.net/), or [here](http://search.maven.org/#search%7Cga%7C1%7Ca%3A%22netlib-native_system-win-x86_64%22)

##### ATLAS

###### Linux

**Ubuntu**
An apt package is available for ATLAS on Ubuntu:

```
sudo apt-get install libatlas-base-dev libatlas-dev
```

**CentOS**
You can install ATLAS on CentOS using:

```
sudo yum install atlas-devel
```

###### OS X

Installing ATLAS on OS X is a somewhat complicated and lengthy process. However, the following commands will work on most machines:

```
wget http://hivelocity.dl.sourceforge.net/project/math-atlas/Stable/3.10.1/atlas3.10.1.tar.bz2 (Download)
tar jxf atlas3.10.1.tar.bz2
mkdir atlas (Creating a directory for ATLAS)
mv ATLAS atlas/src-3.10.1
cd atlas/src-3.10.1
wget http://www.netlib.org/lapack/lapack-3.5.0.tgz (It may be possible that the atlas download already contains this file in which case this command is not needed)
mkdir intel(Creating a build directory)
cd intel
cpufreq-selector -g performance (This command requires root access. It is recommended but not essential)
../configure --prefix=/path to the directory where you want ATLAS installed/ --shared --with-netlib-lapack-tarfile=../lapack-3.5.0.tgz
make
make check
make ptcheck
make time
make install
```

#### GPU

Detailed instructions for installing GPU architectures such as Jcublas can be found [here](http://nd4j.org/gpu_native_backends.html).

## Downloading and installing

## OS X & Linux

### Checking ENV

Before running the DL4J stack build script, you must ensure certain environment variables are defined before running your build. These are outlined below depending on your architecture.

#### LIBND4J_HOME

You will need to know the exact path of the directory where you are running the DL4J build script (you are encouraged to use a clean empty directory). Otherwise, your build will fail. Once you determine this path, add `/libnd4j` to the end of that path and export it to your local environment. This will look like:

```
export LIBND4J_HOME=/home/user/directory/libnd4j
```

#### CPU architecture w/ MKL

You can link with MKL either at build time, or at runtime with binaries initially linked with another BLAS implementation such as OpenBLAS. To build against MKL, simply add the path containing `libmkl_rt.so` (or `mkl_rt.dll` on Windows), say `/path/to/intel64/lib/`, to the `LD_LIBRARY_PATH` environment variable on Linux (or `PATH` on Windows) and build like before. On Linux though, to make sure it uses the correct version of OpenMP, we also might need to set these environment variables:

```bash
export MKL_THREADING_LAYER=GNU
export LD_PRELOAD=/lib64/libgomp.so.1
```

When libnd4j cannot be rebuilt, we can use the MKL libraries after the facts and get them loaded instead of OpenBLAS at runtime, but things are a bit trickier. Please additionally follow the instructions below.

1. Make sure that files such as `/lib64/libopenblas.so.0` and `/lib64/libblas.so.3` are not available (or appear after in the `PATH` on Windows), or they will get loaded by libnd4j by their absolute paths, before anything else.
2. Inside `/path/to/intel64/lib/`, create a symbolic link or copy of `libmkl_rt.so` (or `mkl_rt.dll` on Windows) to the name that libnd4j expect to load, for example:

    ```bash
    ln -s libmkl_rt.so libopenblas.so.0
    ln -s libmkl_rt.so libblas.so.3
    ```

    ```cmd
    copy mkl_rt.dll libopenblas.dll
    copy mkl_rt.dll libblas3.dll
    ```
3. Finally, add `/path/to/intel64/lib/` to the `LD_LIBRARY_PATH` environment variable (or early in the `PATH` on Windows) and run your Java application as usual.


### Build Script

A community-provided script [build-dl4j-stack.sh](https://gist.github.com/crockpotveggies/9948a365c2d45adcf96642db336e7df1) written in bash is available that clones the DL4J stack, builds each repository, and installs them locally to Maven. This script will work on both Linux and OS X platforms.

## Using local dependencies

Once you've installed the DL4J stack to your local maven repository, you can now include it in your build tool's dependencies. Follow the typical [Getting Started](http://deeplearning4j.org/gettingstarted) instructions for Deeplearning4j, and appropriately replace versions with the SNAPSHOT version currently on the [master POM](https://github.com/deeplearning4j/deeplearning4j/blob/master/pom.xml).

Note that some build tools such as Gradle and SBT don't properly pull in platform-specific binaries. You can follow instructions [here](http://nd4j.org/dependencies.html) for setting up your favorite build tool.

## Support and Assistance

If you encounter issues while building locally, the Deeplearning4j [Early Adopters Channel](https://gitter.im/deeplearning4j/deeplearning4j/earlyadopters) is a channel dedicated to assisting with build issues and other source problems. Please reach out on Gitter for help.