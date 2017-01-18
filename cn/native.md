---
title: Native CPU Optimization on Deeplearning4j
layout: default
---

# Native CPU Optimization on Deeplearning4j & ND4J

There are a few ways to tune or improve performance for CPU systems on DL4J and ND4J, which this guide will address. First, let’s define some terminology:

## OpenMP

[OpenMP](http://openmp.org/wp/) is open API for parallel programming for C/C++/Fortran, and since ND4j uses a backend written in C++, we use OpenMP for better parallel performance on CPUs.

## CPU vs. Core vs. HyperThreading

A CPU is a single physical unit that usually consists of multiple cores. Each core is able to process instructions independent of other cores. And each core is subject for HyperThreading - in such systems, that's shown as an additional set of cores. 

Say you have Intel i7-4790 CPU installed: It’s one physical CPU, four physical cores and eight total threads. 

Or you have a dual Intel® Xeon® Processor E5-2683 v4 system: it’s two physical cpus, 16 physical cores each (total of 32 physical cores), 32 virtual cores (total of 64 cores). AMD has the similar architecture, but with slightly different naming, they have CPU -> module -> core gradation, but general idea is still the same there.

## SIMD

That’s an acronym for *Single Instructions Multiple Data*, a kind of parallelism that's supposed to apply some specific instruction to the different array/vector elements in parallel. We use SIMD parallelism mostly for inner loops, or for loops that are too small to exploit OpenMP.

## Parallelism vs. Performance:

It’s quite possible to have a system so powerful it becomes a bottleneck for itself if you don’t limit parallelism. Imagine you have a 20-core CPU, but your task is just a linear operation over a 256-element array. In this case, launching even one parallel thread isn’t worth it, because the same operation will be applied faster with a single thread + SIMD lanes of your CPU, without any overhead produced by launching & operating a new thread. 

That’s why each operation is evaluated for its maximum effective parallelism before launching additional threads.

Now, with terminology and basics defined, let’s discuss performance tuning.

# Performance Tuning

## OMP_NUM_THREADS

`OMP_NUM_THREADS` environment variable defines, how many OpenMP threads will be used for BLAS calls and other native calls. Nd4j tries to guess best possible value for this parameter, but in some cases custom value for this parameter might give better performance. General rule for «best value» here is: number of physical cores for 1 CPU. But please note, this parameter defines maximum number, and actual number of threads launched for any individual op might (and probably will) be lower then this value.

I.e. if you have dual-cpu 8-core system with HyperThreading enabled, total number of cores reported in system will be 32, and best value for `OMP_NUM_THREADS` will be 8.
If you have quad-cpu 10 core system with HyperThreading enabled, total number of cores reported in system will be 80, and best value for `OMP_NUM_THREADS` will be 10.
If you have single cpu with 16 cores and HyperThreading enabled, total number of cores reported in system will be 32, and best value for `OMP_NUM_THREADS` will be somewhere between 8 and 16, depending on workload size.

## Intel MKL

By default, OpenBLAS is used with ND4J, but there’s an option to use the first-class high-performance Intel MKL library together with ND4J/DL4J. That happens automatically, if you have MKL libraries on `$PATH`, they will be used for BLAS operations. [Intel MKL](https://software.intel.com/sites/campaigns/nest/) is available for Linux, OSX and Windows. It’s also possible to compile ND4J with Intel MKL pre-linked, if you have Intel MKL installed. MKL is available free of charge under community licensing Intel. 

## Spark environment

For a distributed environment, some customizations are helpful. 

First of all, you should consider configuring the number of executors per node to the value that excludes HyperThreading cores. That will also save you some space for internal multithreading in native operations. 

Additionally, consider the `OMP_NUM_THREADS` value in such environments. Since you’re going to have a high number of concurrent Java threads provided by Spark, native parallelism should be reduced to something like 2-4 threads. 

So, `OMP_NUM_THREADS=4` might be a good value to start performance tuning. 

## Building from source

Please note: manual compilation requires some skills & understanding in C/C++ field. But when you’re building things from source, and you know your environment, you have a few additional options that provide better performance:

* Vectorized math libraries - using those, you’ll enable better use of CPU SIMD abilities for math functions, widely used in machine learning.
* libsvml available with Intel MKL
* libmvec available on Linux with glibc 2.22+
* -march=native

This is generic compilation optimization, which enables code compilation for your current hardware architecture. On modern processors that usually gives better vectorization.

## Troubleshooting the CPU backend

### ND4J_SKIP_BLAS_THREADS

If you have an unusual BLAS environment, or you have troubles around the `Nd4jBlas.setMaxThreads()` call - set the environment variable  `ND4J_SKIP_BLAS_THREADS` to any value. In this case, the method won’t be triggered. But in this case you’ll also have to set the  `OMP_NUM_THREADS` variable manually as well.

*By Vyacheslav Kokorin*

## Further Reading

* [Optimizing Deeplearning4j on Multiple GPUs](./gpu)
* [Distributed Deep Learning on Spark With DL4J](./spark)
