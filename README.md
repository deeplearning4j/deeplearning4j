# LibND4J

Native operations for nd4j. Build using cmake

Pre requisites:

Gcc 4+ or clang
Cuda (if needed)

A blas implementation or openblas is required

OS specific:


OSX:

Run ./setuposx.sh (ensure you have brew installed)


Linux:

Depends on the distro - ask in the earlyadopters channel for specifics
on distro

Ubuntu Linux 16.04
* sudo apt install libopenblas-dev
* sudo apt install cmake

The standard development headers are needed.

Windows:
see windows.md


1. Set a LIBND4J_HOME as an environment variable to the libnd4j folder you've obtained from GIT

This is required for building nd4j as well.

2. For cpu followed by gpu run:
     
       ./buildnativeoperations.sh blas cpu
       ./buildnativeoperations.sh blas cuda
       
For Debug builds:

    ./buildnativeoperations.sh blas cpu Debug
    ./buildnativeoperations.sh blas cuda Debug


For release builds (default):

    ./buildnativeoperations.sh blas cpu Release
    ./buildnativeoperations.sh blas cuda Release
