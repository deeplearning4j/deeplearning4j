---
title: Using Deeplearning4j with cuDNN
layout: default
---

# Using Deeplearning4j with cuDNN

Deeplearning4j supports CUDA but can be further accelerated with cuDNN, starting with version 0.4.0. Currently only CNNs are supported, but support for RNNs is also planned. To use cuDNN, you will first need to switch ND4J to the CUDA backend. This can be done by replacing `nd4j-native` with `nd4j-cuda-7.5` in your `pom.xml` files, ideally adding a dependency on `nd4j-cuda-7.5-platform` to include automatically binaries from all platforms:

	 <dependency>
	   <groupId>org.nd4j</groupId>
	   <artifactId>nd4j-cuda-7.5-platform</artifactId>
	   <version>${nd4j.version}</version>
	 </dependency>

More information about that can be found among the [installation instructions for ND4J](http://nd4j.org/getstarted).

The only other thing we need to do to have DL4J load cuDNN is to add a dependency on `deeplearning-cuda-7.5`, for example:

	 <dependency>
	   <groupId>org.deeplearning4j</groupId>
	   <artifactId>deeplearning4j-cuda-7.5</artifactId>
	   <version>${dl4j.version}</version>
	 </dependency>

The actual library for cuDNN is not bundled, so be sure to download and install the appropriate package for your platform from NVIDIA:

* [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

Note that only cuDNN 5 is supported. To install, simply extract the library to a directory found in the system path used by native libraries. The easiest way is to place it alongside other libraries from CUDA in the default directory (`/usr/local/lib64/` on Linux, `/usr/local/lib/` on Mac OS X, and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\` on Windows).


