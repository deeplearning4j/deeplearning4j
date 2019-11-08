---
title: Using Deeplearning4j with cuDNN
short_title: cuDNN
description: Using the NVIDIA cuDNN library with DL4J.
category: Configuration
weight: 3
---

## Using Deeplearning4j with cuDNN

Deeplearning4j supports CUDA but can be further accelerated with cuDNN. Most 2D CNN layers (such as ConvolutionLayer, SubsamplingLayer, etc), and also LSTM and BatchNormalization layers support CuDNN.

The only thing we need to do to have DL4J load cuDNN is to add a dependency on `deeplearning4j-cuda-9.2`, `deeplearning4j-cuda-10.0`, or `deeplearning4j-cuda-10.1`, for example:

```xml
<dependency>
	<groupId>org.deeplearning4j</groupId>
	<artifactId>deeplearning4j-cuda-9.2</artifactId>
	<version>{{page.version}}</version>
</dependency>
```

or
```xml
<dependency>
	<groupId>org.deeplearning4j</groupId>
	<artifactId>deeplearning4j-cuda-10.0</artifactId>
	<version>{{page.version}}</version>
</dependency>
```

or
```xml
<dependency>
	<groupId>org.deeplearning4j</groupId>
	<artifactId>deeplearning4j-cuda-10.1</artifactId>
	<version>{{page.version}}</version>
</dependency>
```

The actual library for cuDNN is not bundled, so be sure to download and install the appropriate package for your platform from NVIDIA:

* [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)

Note there are multiple combinations of cuDNN and CUDA supported. At this time the following combinations are supported by Deeplearning4j:
<table style="width:60%">
	<tr>
		<th>CUDA Version</th>
		<th>cuDNN Version</th>
	</tr>
	<tr><td>9.2</td><td>7.2</td></tr>
	<tr><td>10.0</td><td>7.4</td></tr>
	<tr><td>10.1</td><td>7.6</td></tr>
</table>

 
 To install, simply extract the library to a directory found in the system path used by native libraries. The easiest way is to place it alongside other libraries from CUDA in the default directory (`/usr/local/cuda/lib64/` on Linux, `/usr/local/cuda/lib/` on Mac OS X, and `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\bin\`, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\`, or `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin\` on Windows).

Alternatively, in the case of CUDA 10.1, cuDNN comes bundled with the "redist" package of the [JavaCPP Presets for CUDA](https://github.com/bytedeco/javacpp-presets/tree/master/cuda). [After agreeing to the license](https://github.com/bytedeco/javacpp-presets/tree/master/cuda#license-agreements), we can add the following dependencies instead of installing CUDA and cuDNN:
		 
	 <dependency>
	     <groupId>org.bytedeco</groupId>
	     <artifactId>cuda</artifactId>
	     <version>10.1-7.6-1.5.2</version>
	     <classifier>linux-x86_64-redist</classifier>
	 </dependency>
	 <dependency>
	     <groupId>org.bytedeco</groupId>
	     <artifactId>cuda</artifactId>
	     <version>10.1-7.6-1.5.2</version>
	     <classifier>linux-ppc64le-redist</classifier>
	 </dependency>
	 <dependency>
	     <groupId>org.bytedeco</groupId>
	     <artifactId>cuda</artifactId>
	     <version>10.1-7.6-1.5.2</version>
	     <classifier>macosx-x86_64-redist</classifier>
	 </dependency>
	 <dependency>
	     <groupId>org.bytedeco</groupId>
	     <artifactId>cuda</artifactId>
	     <version>10.1-7.6-1.5.2</version>
	     <classifier>windows-x86_64-redist</classifier>
	 </dependency>

Also note that, by default, Deeplearning4j will use the fastest algorithms available according to cuDNN, but memory usage may be excessive, causing strange launch errors. When this happens, try to reduce memory usage by using the [`NO_WORKSPACE` mode settable via the network configuration](/api/{{page.version}}/org/deeplearning4j/nn/conf/layers/ConvolutionLayer.Builder.html#cudnnAlgoMode-org.deeplearning4j.nn.conf.layers.ConvolutionLayer.AlgoMode-), instead of the default of `ConvolutionLayer.AlgoMode.PREFER_FASTEST`, for example:

```java
    // for the whole network
    new NeuralNetConfiguration.Builder()
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
            // ...
    // or separately for each layer
    new ConvolutionLayer.Builder(h, w)
            .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
            // ...

```
