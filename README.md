# LibND4J

Native operations for nd4j. Build using cmake

## Prerequisites

* Gcc 4+ or clang
* Cuda (if needed)
* CMake
* A blas implementation or openblas is required

## OS Specific Requirements

### OSX

Run ./setuposx.sh (ensure you have brew installed)

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
./buildnativeoperations.sh blas cpu Debug
./buildnativeoperations.sh blas cuda Debug
```
#### Ubuntu Linux 16.04

```bash
sudo apt install libopenblas-dev
sudo apt install cmake
Installation of CUDA currently not supported by NVIDIA, working on a fix... 
```

The standard development headers are needed.

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

We can link with MKL either at build time, or at runtime with binaries initially linked with another BLAS implementation such as OpenBLAS. To build against MKL, simply add the path containing `libmkl_rt.so` (or `mkl_rt.dll` on Windows), say `/path/to/intel64/lib/`, to the `LD_LIBRARY_PATH` environment variable on Linux (or `PATH` on Windows) and build like before. On Linux though, to make sure it uses the correct version of OpenMP, we also might need to set these environment variables:

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

