# LibND4J

Native operations for nd4j. Build using cmake

## Prerequisites

* Gcc 4+ or clang
* Cuda (if needed)
* CMake
* A blas implementation or openblas is required

## OS Specific Requirements

### Android

[Download the NDK](https://developer.android.com/ndk/downloads/), extract it somewhere, and execute the following commands, replacing `android-xxx` with either `android-arm` or `android-x86`:

```bash
git clone https://github.com/bytedeco/javacpp-presets
git clone https://github.com/deeplearning4j/libnd4j
git clone https://github.com/deeplearning4j/nd4j
export ANDROID_NDK=/path/to/android-ndk/
export LIBND4J_HOME=$PWD/libnd4j/
export OpenBLAS_HOME=$PWD/javacpp-presets/openblas/cppbuild/android-xxx/
cd javacpp-presets/openblas
bash cppbuild.sh install -platform android-xxx
cd ../../libnd4j
bash buildnativeoperations.sh -platform android-xxx
cd ../nd4j
mvn clean install -Djavacpp.platform=android-xxx -DskipTests -pl '!nd4j-backends/nd4j-backend-impls/nd4j-cuda,!nd4j-backends/nd4j-backend-impls/nd4j-cuda-platform'
```

### OSX

Run ./setuposx.sh (Please ensure you have brew installed)

See [macOSx10 (CPU only).md](macOSx10 (CPU only).md)

### Linux

Depends on the distro - ask in the earlyadopters channel for specifics
on distro

#### Ubuntu Linux 15.10

```bash
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1504-7-5-local_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1504-7-5-local_7.5-18_amd64.deb
sudo apt-get update
sudo apt-get install cuda
sudo apt-get install libopenblas-dev
sudo apt-get install cmake
sudo apt-get install gcc-4.9
sudo apt-get install g++-4.9
sudo apt-get install git
git clone https://github.com/deeplearning4j/libnd4j
cd libnd4j/
export LIBND4J_HOME=~/libnd4j/
sudo rm /usr/bin/gcc
sudo rm /usr/bin/g++
sudo ln -s /usr/bin/gcc-4.9 /usr/bin/gcc
sudo ln -s /usr/bin/g++-4.9 /usr/bin/g++
./buildnativeoperations.sh
./buildnativeoperations.sh -c cuda
```
#### Ubuntu Linux 16.04

```bash
sudo apt install libopenblas-dev
sudo apt install cmake
sudo apt install nvidia-cuda-dev nvidia-cuda-toolkit nvidia-361
export TRICK_NVCC=YES
./buildnativeoperations.sh
./buildnativeoperations.sh -c cuda

```

The standard development headers are needed.

#### CentOS 6

```bash
yum install centos-release-scl-rh epel-release
yum install devtoolset-3-toolchain maven30 cmake3 git openblas-devel
scl enable devtoolset-3 maven30 bash
./buildnativeoperations.sh
./buildnativeoperations.sh -c cuda
```

### Windows

See [Windows.md](windows.md)

## Setup for All OS

1. Set a LIBND4J_HOME as an environment variable to the libnd4j folder you've obtained from GIT
     *  Note: this is required for building nd4j as well.

2. Setup cpu followed by gpu, run the following on the command line:
     * For standard builds:
    
        ```bash
        ./buildnativeoperations.sh
        ./buildnativeoperations.sh -c cuda
        ```
        
     * For Debug builds:
     
        ```bash
        ./buildnativeoperations.sh blas -b debug
        ./buildnativeoperations.sh blas -c cuda -b debug
        ```
        
     * For release builds (default):
     
        ```bash
        ./buildnativeoperations.sh
        ./buildnativeoperations.sh -c cuda
        ```

## Linking with MKL

We can link with MKL either at build time, or at runtime with binaries initially linked with another BLAS implementation such as OpenBLAS. In either case, simply add the path containing `libmkl_rt.so` (or `mkl_rt.dll` on Windows), say `/path/to/intel64/lib/`, to the `LD_LIBRARY_PATH` environment variable on Linux (or `PATH` on Windows), and build or run your Java application as usual. If you get an error message like `undefined symbol: omp_get_num_procs`, it probably means that `libiomp5.so`, `libiomp5.dylib`, or `libiomp5md.dll` is not present on your system. In that case though, it is still possible to use the GNU version of OpenMP by setting these environment variables on Linux, for example:

```bash
export MKL_THREADING_LAYER=GNU
export LD_PRELOAD=/usr/lib64/libgomp.so.1
```

