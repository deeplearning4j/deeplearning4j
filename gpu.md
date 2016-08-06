---
title: Deeplearning4j With GPUs
layout: default
---

# Deeplearning4j With GPUs

Deeplearning4j works on distributed GPUs, as well as on native. We allow users to run locally on a single GPU such as the NVIDIA Tesla, Titan or GeForce GTX, and in the cloud on NVIDIA GRID GPUs. 

In order to train a neural network on GPUs, you need to make a single change your POM.xml file. In the [Quickstart](./quickstart), you'll find a POM file configured to run on CPUs by default. It looks like this:

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-examples/blob/master/pom.xml?slice=52:62"></script>

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

## Troubleshooting

If you have several GPUs, but your system is forcing you to use just one, there's a solution. Just add `CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);` as first line of your `main()` method.

<p align="center">
<a href="./quickstart" class="btn btn-custom" onClick="ga('send', 'event', ‘quickstart', 'click');">Get Started With Deeplearning4j on GPUs</a>
</p>


## Multi-GPU data parallelism

If your system has multiple GPUs installed, you can train your model in data-parallel mode. We have simple wrapper available for that:

Consider using something like this:

        ParallelWrapper wrapper = new ParallelWrapper.Builder(YourExistingModel)
            .prefetchBuffer(24)
            .workers(4)
            .averagingFrequency(1)
            .reportScoreAfterAveraging(true)
            .useLegacyAveraging(false)
            .build();

ParallelWrapper takes your existing model as primary argument, and does training in parallel. In the case of GPUs, it’s worth keeping the number of workers equal to or higher than number of GPUs. Exact values are subject to tuning, since they depend on your task as well as the hardware available.

Within `ParallelWrapper`, your initial model will be duplicated, and each worker will be training its own model. After every *X* iterations, defined by `averagingFrequency(X)`, all models will be averaged, and training will continue after that. 

It's worth noting that for data-parallel training, a higher learning rate is recommended. Something around +20% should be a good starting value.

## HALF Datatype

If your app can afford using half-precision math (typically neural nets can afford this), you can enable this as data type for your app, and you'll see following benefits:

* 2x less GPU ram used
* up to 200% performance gains on memory-intensive operations, though the actual performance boost depends on the task and hardware used.

        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);

Place this call as the first line of your app, so that all subsequent allocations/calculations will be done using the HALF data type. 

## Larger Grids

For most GPUs, default values are fine, but if you’re using high-end hardware and your data is massive enough, it might be worth trying bigger grid/block limits. Something like this might be used:

    CudaEnvironment.getInstance().getConfiguration()
          .setMaximumGridSize(512)
          .setMaximumBlockSize(512);

This won’t force all, even minor operations, to use specified grid dimensions, but it’ll create theoretical limits for them. 

## Allow for a larger cache

Since ND4J is based on JAva, the cache size is very important for CUDA backend, and it’s able to dramatically increase or decrease performance. If you have plenty of RAM - just allow larger caches.

Something like this might be used:

        CudaEnvironment.getInstance().getConfiguration()
        .setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
        .setMaximumDeviceCache(6L * 1024 * 1024 * 1024L)
        .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
        .setMaximumHostCache(6L * 1024 * 1024 * 1024L);

This code will allow to cache up to 6GB of GPU RAM (it doesn’t mean that it WILL allocate that much though), and each individually cached memory chunk for both host and GPU memory might be up to 1GB in size. 

Since the cache in Nd4j works has a «reuse» paradigm, such high values don’t mean anything bad. Only memory chunks that were allocated for your app might be cached for future reuse.
