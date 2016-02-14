---
layout: default
title: Getting Started on Windows
---

# Getting Started on Windows

* To run examples, go to our [quickstart](../quickstart.html). 

* While our Windows install is not always easy, Deeplearning4j is one of the few open-source deep learning projects that actually cares about trying to support the Windows community. Please see the [Windows section of our ND4J page](http://nd4j.org/getstarted.html#windows) for more instructions. 

* Install [MinGW 32 bits](http://www.mingw.org/) even if you have a 64-bit computer (the download button is on the upper right), and then download the [Prebuilt dynamic libraries using Mingw](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw). 

* Install [Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/). (Lapack will ask if you have Intel compilers. You do not.)

* Lapack offers the alternative of [VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke). You'll also want to look at the documentation for [Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/). 

* Alternatively, you can bypass MinGW and copy the Blas dll files to a folder in your PATH. For example, the path to the MinGW bin folder is: /usr/x86_64-w64-mingw32/sys-root/mingw/bin. To read more about the PATH variable in Windows, please read the [top answer on this StackOverflow page](https://stackoverflow.com/questions/3402214/windows-7-maven-2-install). 

* Cygwin is not supported. You must install DL4J from **DOS Windows**.  

* Running this file, [WindowsInfo.bat](https://gist.github.com/AlexDBlack/9f70c13726a3904a2100), can help debug your Windows install. Here's one [example of its output](https://gist.github.com/AlexDBlack/4a3995fea6dcd2105c5f) that shows what to expect. First download it, then open a command window / terminal. `cd` to the directory to which it was dowloaded. Enter `WindowsInfo` and hit enter. To copy its output, right click on command window -> select all -> hit enter. Output is then on clipboard.

For OpenBlas (see below) on **Windows**, download this [file](https://www.dropbox.com/s/6p8yn3fcf230rxy/ND4J_Win64_OpenBLAS-v0.2.14.zip?dl=1). Extract to somewhere such as `C:/BLAS`. Add that directory to your system's `PATH` environment variable.

### <a id="open"> OpenBlas </a>

To make sure the native libs on the x86 backend work, you need `/opt/OpenBLAS/lib` on the system path. After that, enter these commands in the prompt

			sudo cp libopenblas.so liblapack.so.3
			sudo cp libopenblas.so libblas.so.3

We added this so that [Spark](http://deeplearning4j.org/spark) would work with OpenBlas.

If OpenBlas is not working correctly, follow these steps:

* Remove Openblas if you installed it.
* Run `sudo apt-get remove libopenblas-base`
* Download the development version of OpenBLAS
* `git clone git://github.com/xianyi/OpenBLAS`
* `cd OpenBLAS`
* `make FC=gfortran`
* `sudo make PREFIX=/usr/local/ install`
* As a last step, restart your IDE. 

* For a complete review of requirements, see our [getting started page](../gettingstarted.html).
