---
title: How to Speed Up Spark With Native Binaries and OpenBlas
layout: default
---

# How to Speed Up Spark With Native Binaries and OpenBlas

Spark is a distributed form of MapReduce. It ranked as the Apache Foundation's most popular open-source project last year. It's supposed to be fast, but because of some licensing issues, it doesn't run fast out of the box. 

This step-by-step guide shows you how to enable native binaries for matrix operations in Spark. Spark can't ship with them, because Spark is licensed under Apache 2.0, while the native binaries of OpenBlas are LGPL. 

According to the [Apache Foundation](https://www.apache.org/legal/resolved.html), "LGPL-licensed works must not be included in Apache products" because of "the restrictions it places on larger works, violating the third license criterion."

Since native binaries cannot be "packaged" with Spark, we either have to turn them on ourselves, or run matrix operations on Spark's default Java configuration, which sadly is not performant. Native binaries will get you up to a 10x acceleration on matrix multiplies. 

That matters a lot to us, because deep learning is computationally expensive -- that's the downside of its [record-breaking accuracy](../accuracy.html) on numerous [use cases](../use_cases.html). So we train neural networks in parallel on Spark using multiple CPUs or GPUs, for things like scene classification.

Instructions will vary slightly from operating system to operating system.

### <a id="open"> OpenBlas for Linux</a>

We need [Spark](http://deeplearning4j.org/spark) to work with [OpenBlas](http://www.openblas.net/), an optimized BLAS library based on GotoBLAS2.

To make sure that the native libs on [ND4J's x86 backend](http://nd4j.org/backend.html) work, you need `/opt/OpenBLAS/lib` on the system path. ([ND4J](http://nd4j.org) is a scientific computing engine for the JVM.)

After that, enter these commands in the prompt:

			sudo cp libopenblas.so liblapack.so.3
			sudo cp libopenblas.so libblas.so.3

If OpenBlas does not work correctly, follow these steps:

* Remove Openblas by running `sudo apt-get remove libopenblas-base`
* Download the development version of OpenBLAS: `git clone git://github.com/xianyi/OpenBLAS`
* `cd OpenBLAS`
* `make FC=gfortran`
* `sudo make PREFIX=/usr/local/ install`
* With **Linux**, double check if the symlinks for `libblas.so.3` and `liblapack.so.3` are present anywhere in your `LD_LIBRARY_PATH`. If they aren't, add the links to `/usr/lib`. A symlink is a "symbolic link." You can set it up like this (the -s makes the link symbolic):

		ln -s TARGET LINK_NAME
		// interpretation: ln -s "to-here" <- "from-here"

* The "from-here" is the symbolic link that does not exist yet, and which you are creating. Here's StackOverflow on [how to create a symlink](https://stackoverflow.com/questions/1951742/how-to-symlink-a-file-in-linux). And here's the [Linux man page](http://linux.die.net/man/1/ln).
* As a last step, restart your IDE. 

### Windows

* For OpenBlas on **Windows**, [download this file](https://www.dropbox.com/s/6p8yn3fcf230rxy/ND4J_Win64_OpenBLAS-v0.2.14.zip?dl=1). 
* Extract the file to a directory such as `C:/BLAS`. 
* Add that directory to your system's `PATH` environment variable.

### Other Tutorials

* [Regression & Neural Networks](../linear-regression.html)
* [Word2vec: Extracting Relations From Raw Text](../word2vec.html)
* [Restricted Boltzmann Machines: The Building Blocks of Deep-Belief Networks](../restrictedboltzmannmachine.html)
* [Recurrent Networks and Long Short-Term Memory Units](../lstm.html)
* [Eigenvectors, PCA, Covariance and Entropy](../eigenvector.html)

<!--
### OSX

Anything different for Mac?

How can people test to make sure everything is working? -->
