---
title: Deeplearning4j With GPUs
layout: default
---

# Deeplearning4j With GPUs

Deeplearning4j works on distributed GPUs, as well as on native. We allow users to run locally on a single GPU such as the NVIDIA Tesla or GeForce GTX, and in the cloud on NVIDIA GRID GPUs. 

In order to train a neural network on GPUs, you need to make a single change your POM.xml file. In the [Quickstart](./quickstart), you'll find a POM file configured to run on CPUs by default. It looks like this:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml?slice=52:62"></script>

You want to make Deeplearning4j run on GPUs, you swap out the `artifactId` line under `nd4j` in your dependencies, replacing `nd4j-native` with `nd4j-cuda-7.5`. That's it...

``` xml
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.nd4j</groupId>
            <artifactId>nd4j-cuda-7.5</artifactId>
            <version>${nd4j.version}</version>
        </dependency>
    </dependencies>
</dependencyManagement>
```

ND4J is the numerical computing engine that powers Deeplearning4j. It has what we call "backends", or different types of hardware that it works on. In the [Deeplearning4j Gitter channel](https://gitter.im/deeplearning4j/deeplearning4j), you'll here people talk about backends, and they're just referring to the packages that point to one chip or another. The backends are where we've done the work of optimizing on the hardware.
