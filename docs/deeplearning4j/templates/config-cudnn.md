---
title: Using Deeplearning4j with cuDNN
short_title: cuDNN
description: Using the NVIDIA cuDNN library with DL4J.
category: Configuration
weight: 3
---

## Using Deeplearning4j with cuDNN

Deeplearning4j supports CUDA but can be further accelerated with cuDNN. Most 2D CNN layers (such as ConvolutionLayer, SubsamplingLayer, etc), and also LSTM and BatchNormalization layers support CuDNN.

To use cuDNN, you will first need to switch ND4J to the CUDA backend. This can be done by replacing `nd4j-native` with `nd4j-cuda-8.0`, `nd4j-cuda-9.0`, or `nd4j-cuda-9.1`  in your `pom.xml` files, ideally adding a dependency on `nd4j-cuda-8.0-platform`, `nd4j-cuda-9.0-platform`, or `nd4j-cuda-9.2-platform` to include automatically binaries from all platforms:

	 <dependency>
	   <groupId>org.nd4j</groupId>
	   <artifactId>nd4j-cuda-8.0-platform</artifactId>
	   <version>{{page.version}}</version>
	 </dependency>

or

	 <dependency>
	   <groupId>org.nd4j</groupId>
	   <artifactId>nd4j-cuda-9.0-platform</artifactId>
	   <version>{{page.version}}</version>
	 </dependency>

or

	 <dependency>
	   <groupId>org.nd4j</groupId>
	   <artifactId>nd4j-cuda-9.2-platform</artifactId>
	   <version>{{page.version}}</version>
	 </dependency>

More information about that can be found among the [installation instructions for ND4J](http://nd4j.org/getstarted).

The only other thing we need to do to have DL4J load cuDNN is to add a dependency on `deeplearning4j-cuda-8.0`, `deeplearning4j-cuda-9.0`, or `deeplearning4j-cuda-9.2`, for example:

	 <dependency>
	   <groupId>org.deeplearning4j</groupId>
	   <artifactId>deeplearning4j-cuda-8.0</artifactId>
	   <version>{{page.version}}</version>
	 </dependency>

or

	 <dependency>
	   <groupId>org.deeplearning4j</groupId>
	   <artifactId>deeplearning4j-cuda-9.0</artifactId>
	   <version>{{page.version}}</version>
	 </dependency>

or

	 <dependency>
	   <groupId>org.deeplearning4j</groupId>
	   <artifactId>deeplearning4j-cuda-9.2</artifactId>
	   <version>{{page.version}}</version>
	 </dependency>

The actual library for cuDNN is not bundled, so be sure to download and install the appropriate package for your platform from NVIDIA:

* [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

Note there are multiple combinations of cuDNN and CUDA supported. At this time the following combinations are supported by Deeplearning4j:
<table style="width:60%">
	<tr>
		<th>CUDA Version</th>
		<th>cuDNN Version</th>
	</tr>
	<tr><td>8.0</td><td>6.0</td></tr>
	<tr><td>9.0</td><td>7.0</td></tr>
	<tr><td>9.2</td><td>7.1</td></tr>
</table>

 
 To install, simply extract the library to a directory found in the system path used by native libraries. The easiest way is to place it alongside other libraries from CUDA in the default directory (`/usr/local/cuda/lib64/` on Linux, `/usr/local/cuda/lib/` on Mac OS X, and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\`, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin\`, or `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\bin\` on Windows).

Also note that, by default, Deeplearning4j will use the fastest algorithms available according to cuDNN, but memory usage may be excessive, causing strange launch errors. When this happens, try to reduce memory usage by using the [`NO_WORKSPACE` mode settable via the network configuration](/api/{{page.version}}/org/deeplearning4j/nn/conf/layers/ConvolutionLayer.Builder.html#cudnnAlgoMode-org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode-), instead of the default of `ConvolutionLayer.AlgoMode.PREFER_FASTEST`, for example:

```java
    // ...
    new ConvolutionLayer.Builder(h, w)
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
            // ...

```