---
title: Using Deeplearning4j with cuDNN
layout: default
---

# Using Deeplearning4j with cuDNN

Deeplearning4j supports CUDA but can be further accelerated with cuDNN. Starting with version 0.9.1, both CNNs and LSTMs are supported. To use cuDNN, you will first need to switch ND4J to the CUDA backend. This can be done by replacing `nd4j-native` with `nd4j-cuda-7.5` or `nd4j-cuda-8.0` in your `pom.xml` files, ideally adding a dependency on `nd4j-cuda-7.5-platform` or `nd4j-cuda-8.0-platform` to include automatically binaries from all platforms:

	 <dependency>
	   <groupId>org.nd4j</groupId>
	   <artifactId>nd4j-cuda-7.5-platform</artifactId>
	   <version>${nd4j.version}</version>
	 </dependency>

or

	 <dependency>
	   <groupId>org.nd4j</groupId>
	   <artifactId>nd4j-cuda-8.0-platform</artifactId>
	   <version>${nd4j.version}</version>
	 </dependency>

More information about that can be found among the [installation instructions for ND4J](http://nd4j.org/getstarted).

The only other thing we need to do to have DL4J load cuDNN is to add a dependency on `deeplearning4j-cuda-7.5` or `deeplearning4j-cuda-8.0`, for example:

	 <dependency>
	   <groupId>org.deeplearning4j</groupId>
	   <artifactId>deeplearning4j-cuda-7.5</artifactId>
	   <version>${dl4j.version}</version>
	 </dependency>

or

	 <dependency>
	   <groupId>org.deeplearning4j</groupId>
	   <artifactId>deeplearning4j-cuda-8.0</artifactId>
	   <version>${dl4j.version}</version>
	 </dependency>

The actual library for cuDNN is not bundled, so be sure to download and install the appropriate package for your platform from NVIDIA:

* [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

Note that only cuDNN 6.0 is supported. To install, simply extract the library to a directory found in the system path used by native libraries. The easiest way is to place it alongside other libraries from CUDA in the default directory (`/usr/local/cuda/lib64/` on Linux, `/usr/local/cuda/lib/` on Mac OS X, and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin\` or `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\` on Windows).

Also note that, by default, Deeplearning4j will use the fastest algorithms available according to cuDNN, but memory usage may be excessive, causing strange launch errors. When this happens, try to reduce memory usage by using the [`NO_WORKSPACE` mode settable via the network configuration](https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/layers/ConvolutionLayer.Builder.html#cudnnAlgoMode-org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode-), instead of the default of `ConvolutionLayer.AlgoMode.PREFER_FASTEST`, for example:

```java
    // ...
    new ConvolutionLayer.Builder(h, w)
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
            // ...

```

