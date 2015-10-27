---
title: 
layout: default
---

# How to Make Spark Fast

Spark is a distributed form of MapReduce. It ranked as the Apache Foundation's most popular open-source project last year. It's supposed to be fast, but because of some licensing issues, it doesn't run fast out of the box. 

This step-by-step guide shows you how to enable native binaries for matrix operations in Spark. Spark can't ship with them, because Spark is licensed under Apache 2.0, while the native binaries of OpenBlas are LGPL. 

According to the Apache Foundation, "LGPL-licensed works must not be included in Apache products" because of "the restrictions it places on larger works, violating the third license criterion." (https://www.apache.org/legal/resolved.html)

Since native binaries cannot be "packaged" with Spark, we either have to turn them on ourselves, or run matrix operations on Spark's default Java configuration, which is not performant. 

Instructions will vary slightly from operating system to operating system.

###Linux

 We'll start with Linux, since many will want to use Spark on servers. You'll want to run these instructions for each worker.

* If you have already installed Openblas, remove it by running `sudo apt-get remove libopenblas-base`
* Next, download the development version of OpenBLAS.
* Run `git clone git://github.com/xianyi/OpenBLAS`
* `cd OpenBLAS`
* `make FC=gfortran`
* `sudo make PREFIX=/usr/local/ install`
* Double check if the symlinks for `libblas.so.3` and `liblapack.so.3` are present anywhere in your `LD_LIBRARY_PATH`.

You need `/opt/OpenBLAS/lib` on the system path. After that, enter these commands in the prompt:

			sudo cp libopenblas.so liblapack.so.3
			sudo cp libopenblas.so libblas.so.3

### Windows

For OpenBlas on **Windows**, download this [file](https://www.dropbox.com/s/6p8yn3fcf230rxy/ND4J_Win64_OpenBLAS-v0.2.14.zip?dl=1). Extract to somewhere such as `C:/BLAS`. Add that directory to your system's `PATH` environment variable.

### OSX

<!--
Anything different for Mac?

How can people test to make sure everything is working? -->
