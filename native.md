---
title: Native CPU Optimization on Deeplearning4j
layout: default
---

# Native CPU Optimization on Deeplearning4j & ND4J

This guide will address a few ways to tune or improve performance for CPU systems on DL4J and ND4J. First, let’s define some terminology:

## OpenMP

[OpenMP](http://openmp.org/wp/) is an open API for parallel programming using C/C++/Fortran. Since ND4j uses a backend written in C++, we use OpenMP for better parallel performance on CPUs.

## CPU vs. Core vs. HyperThreading

A CPU is a single physical unit that usually consists of multiple cores. Each core is able to process instructions independent of other cores. Each core is subject to HyperThreading and in such systems, that is shown as an additional set of cores. 

For example, let's say you have an Intel i7-4790 CPU installed. It’s one physical CPU, four physical cores and eight total threads. 

Perhaps you have a dual Intel® Xeon® Processor E5-2683 v4 system. It’s two physical CPUs, 16 physical cores each (total of 32 physical cores) and 32 virtual cores (total of 64 cores). AMD has a similar architecture, but with a slightly different naming convention. They have CPU -> Module -> Core Gradation, but the general idea is the same. 

## SIMD

That’s an acronym for *Single Instructions Multiple Data*, a kind of parallelism that is supposed to apply some specific instruction to the different array/vector elements in parallel. We use SIMD parallelism mostly for inner loops, or for loops that are too small to exploit OpenMP.

## Parallelism vs. Performance

It’s quite possible to have a system so powerful that it impedes itself if you don’t limit parallelism. Imagine that you have a 20-core CPU, but your task is just a linear operation over a 256-element array. In this case, launching even one parallel thread isn’t worth it. The reason being that the same operation will be applied faster with a single thread + SIMD lanes of your CPU, without any overhead produced by launching & operating a new thread. 

That’s why each operation is evaluated for its maximum effective parallelism before launching additional threads.

Now, with terminology and basics defined, let’s discuss performance tuning.

# Performance Tuning

## OMP_NUM_THREADS

`OMP_NUM_THREADS` environment variable defines how many OpenMP threads will be used for BLAS calls and other native calls. ND4J tries to guess the best possible value for this parameter but in some cases, custom values for this parameter might perform better. The general rule for the «best value» here is the number of physical cores for 1 CPU. However, please note this parameter defines the maximum number and actual number of threads launched. For any individual op might (and probably will) be lower then this value.

I.e. If you have a dual-cpu 8-core system with HyperThreading enabled, the total number of cores reported in the system will be 32, and the best value for `OMP_NUM_THREADS` will be 8.
If you have a quad-cpu 10 core system with HyperThreading enabled, the total number of cores reported in the system will be 80, and the best value for `OMP_NUM_THREADS` will be 10.
If you have a single cpu with 16 cores and HyperThreading enabled, the total number of cores reported in the system will be 32, and the best value for `OMP_NUM_THREADS` will be somewhere between 8 and 16, depending on workload size.

## Parallelism thresholds
If you think that you might get better performance with your specific CPU, i.e. if your CPU supports AVX-512 instructions set, you could try to change parallelism thresholds for different operation types. We've exposed few special methods for that:

```java
NativeOpsHolder.getInstance().getDeviceNativeOps().setElementThreshold(16384)
NativeOpsHolder.getInstance().getDeviceNativeOps().setTADThreshold(64)
```

**.setElementThreshold()** call allows you to specify the number of array elements processed by one OpenMP thread. I.e. if you have an AVX-512-capable CPU, you might want to keep that value high enough to avoid spawning threads and use SIMD instead.
 
**.setTADThreshold()** has a similar effect. It allows you to specify the number of tensors (TADs) processed by single OpenMP thread. Depending on CPU model (and amount of CPU cache) you might want to either raise this value or lower it. 


## Intel MKL

By default, OpenBLAS is used with ND4J, but there’s an option to use the first-class high-performance Intel MKL library together with ND4J/DL4J. That happens automatically. If you have MKL libraries on `$PATH`, they will be used for BLAS operations. [Intel MKL](https://software.intel.com/sites/campaigns/nest/) is available for Linux, OSX and Windows. It’s also possible to compile ND4J with Intel MKL pre-linked if you have Intel MKL installed. MKL is available free of charge under community licensing Intel. 

## Spark environment

For a distributed environment, some customizations are helpful. 

First of all, you should consider configuring the number of executors per node to the value that excludes HyperThreading cores. That will also save you some space for internal multithreading in native operations. 

Additionally, consider the `OMP_NUM_THREADS` value in such environments. Since you’re going to have a high number of concurrent Java threads provided by Spark, native parallelism should be reduced to something like 2-4 threads. 

So, `OMP_NUM_THREADS=4` might be a good value to start performance tuning. 

## Building from source

Please note: manual compilation requires some skills & understanding in C/C++ field. But when you’re building things from source, and you know your environment, you have a few additional options that will provide better performance:

* Vectorized math libraries - using these, you’ll get better use of CPU SIMD abilities for math functions, widely used in machine learning.
* libsvml available with Intel MKL
* libmvec available on Linux with glibc 2.22+
* -march=native

This is a generic compilation optimization, which enables code compilation for your current hardware architecture. On modern processors, that usually gives better vectorization.

## Troubleshooting the CPU backend

### ND4J_SKIP_BLAS_THREADS

If you have an unusual BLAS environment, or you have trouble around the `Nd4jBlas.setMaxThreads()` call - set the environment variable `ND4J_SKIP_BLAS_THREADS` to any value. In this case, the method won’t be triggered but you’ll also have to set the `OMP_NUM_THREADS` variable manually as well.


### Fallback mode
Recently, we’ve discovered that on some platforms, popular BLAS-libraries can be unstable, causing crashes under different circumstances. To address that (and possible future issues as well), we’ve provided optional «fallback mode», which will cause ND4J to use in-house solutions as a workaround for possible issues. It acts as a «safe mode», well known for any modern-OS user.

To activate fallback mode you only need to set special environment variable: **ND4J_FALLBACK**. Set it to «**true**» or to **1** before launching your app. It’s possible to use this variable in an Apache Spark environment, as well as in a standalone app.


## How it works after all?

Native backend for ND4J is built with C++ and uses OpenMP internally. Basic idea is implicit parallelism: single JVM thread turns into variable number of threads used during Op invocation. 

This gives us simplified process & memory management in Java (i.e. you're always sure you have single thread accessing given INDArray instance) and at the same time Ops are using OpenMP threads + SIMD optimized loops for better performance.

We use two kinds of internal parallelism:
- Element-level parallelism: Each element in INDArray is processed by separate OpenMP thread or SIMD lane.
- TAD-level parallelism: Each OpenMP thread processes its own tensor within original operand.
 



*By Vyacheslav Kokorin*

## Further Reading

* [Optimizing Deeplearning4j on Multiple GPUs](./gpu)
* [Distributed Deep Learning on Spark With DL4J](./spark)
