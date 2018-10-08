---
title: Deeplearning4j Hardware and CPU/GPU Setup
short_title: GPU/CPU Setup
description: Hardware setup for Eclipse Deeplearning4j, including GPUs and CUDA.
category: Configuration
weight: 1
---

## ND4J backends for GPUs and CPUs

You can choose GPUs or native CPUs for your backend linear algebra operations by changing the dependencies in ND4J's POM.xml file. Your selection will affect both ND4J and DL4J being used in your application.

If you have CUDA v9.2+ installed and NVIDIA-compatible hardware, then your dependency declaration will look like:

```xml
<dependency>
 <groupId>org.nd4j</groupId>
 <artifactId>nd4j-cuda-{{ page.cudaVersion }}</artifactId>
 <version>{{ page.version }}</version>
</dependency>
```

Otherwise you will need to use the native implementation of ND4J as a CPU backend:

```xml
<dependency>
 <groupId>org.nd4j</groupId>
 <artifactId>nd4j-native</artifactId>
 <version>{{ page.version }}</version>
</dependency>
```

## System architectures

If you are developing your project on multiple operating systems/system architectures, you can add `-platform` to the end of your `artifactId` which will download binaries for most major systems.

```xml
<dependency>
 ...
 <artifactId>nd4j-native-platform</artifactId>
 ...
</dependency>
```

## Multiple GPUs

If you have several GPUs, but your system is forcing you to use just one, you can use the helper `CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);` as first line of your `main()` method.

## CuDNN

See our page on [CuDNN](./deeplearning4j-config-cudnn).


## CUDA Installation

Check the NVIDIA guides for instructions on setting up CUDA on the NVIDIA [website](http://docs.nvidia.com/cuda/).