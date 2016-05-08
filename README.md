# LibND4J

Native operations for nd4j. Build using cmake

**Prerequisites:**

* Gcc 4+ or clang
* Cuda (if needed)
* CMake
* A blas implementation or openblas is required

----

**OS Specific Requirements:**


*OSX:*

Run ./setuposx.sh (ensure you have brew installed)


*Linux:*

Depends on the distro - ask in the earlyadopters channel for specifics
on distro

*Ubuntu Linux 15.10*
> wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1504-7-5-local_7.5-18_amd64.deb

> sudo dpkg -i cuda-repo-ubuntu1504-7-5-local_7.5-18_amd64.deb

> sudo apt-get update

> sudo apt-get install cuda

> sudo apt-get install libopenblas-dev

> sudo apt-get install install cmake

> sudo apt-get install gcc-4.9

> sudo apt install g++-4.9

> sudo apt-get install git

> git clone https://github.com/deeplearning4j/libnd4j

> cd libnd4j/

> export LIBND4J_HOME=~/libnd4j/

> sudo rm /usr/bin/gcc

> sudo rm /usr/bin/g++

> sudo ln -s /usr/bin/gcc-4.9 /usr/bin/gcc

> sudo ln -s /usr/bin/g++-4.9 /usr/bin/g++

> ./buildnativeoperations.sh blas cpu Debug

> ./buildnativeoperations.sh blas cuda Debug

*Ubuntu Linux 16.04* 
* sudo apt install libopenblas-dev
* sudo apt install cmake
* Installation of CUDA currently not supported by NVIDIA, working on a fix... 

The standard development headers are needed.

*Windows:*
See windows.md

------

**Setup for All OS:**

1. Set a LIBND4J_HOME as an environment variable to the libnd4j folder you've obtained from GIT
     *  Note: this is required for building nd4j as well.

2. Setup cpu followed by gpu, run the following on the command line:
     * For standard builds:

               ./buildnativeoperations.sh
               ./buildnativeoperations.sh -c cuda

     * For Debug builds:

               ./buildnativeoperations.sh blas -b debug
               ./buildnativeoperations.sh blas -c cuda -b debug

     * For release builds (default):

               ./buildnativeoperations.sh
               ./buildnativeoperations.sh -c cuda


