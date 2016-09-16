---
title: Native CPU Optimization on Deeplearning4j
layout: default
---

# Native CPU Optimization on Deeplearning4j

With DL4j/ND4j you have few ways to tune or improve your performance for CPU systems, and this guide should help you with them. But before that, let’s define some terminology used in this guide:

## OpenMP

OpenMP is open API for parallel programming for C/C++/Fortran, and since ND4j uses backend written in C++, we use OpenMP for better parallel performance on CPU.

## CPU vs Core vs HyperThreading:

CPU is single physical unit, that usually consists of multiple cores. Each core is able to process instructions independent of other cores. And on top of that, each core is subject for HyperThreading - in such systems, it’s shown as additional set of cores. So, say you have Intel i7-4790 CPU installed: It’s physical 1 cpu, physical 4 cores, 8 total threads. Or, say, you have dual Intel® Xeon® Processor E5-2683 v4 system: it’s 2 physical cpus, 16 physical cores each (total of 32 physical cores), 32 virtual cores (total of 64 cores). AMD has the similar architecture, but with slightly different naming, they have CPU -> module -> core gradation, but general idea is still the same there.

## SIMD:

That’s an acronym for Single Instructions Multiple Data, parallelism model, where it’s supposed to apply some specific instruction to the different array/vector elements in parallel. We use SIMD parallelism for inner loops mostly, or for loops that are too small to exploit OpenMP.

## Parallelism vs Performance:

It’s quite possible to get system powerful enough, to become bottleneck for itself, if you don’t limit parallelism. Imagine you have 20-core CPU, but your task is just linear operation over 256-elements array. In this case, launching even one parallel thread isn’t worth it, because the same operation will be applied faster with single thread + SIMD lanes of your CPU, without any overhead produced by launching & operating new thread. That’s why, each operation is evaluated for it’s maximum effective parallelism before launching additional threads.

Now, with terminology & basics defined, let’s discuss performance tuning.

Performance tuning:

## OMP_NUM_THREADS

`OMP_NUM_THREADS` environment variable defines, how many OpenMP threads will be used for BLAS calls and other native calls. Nd4j tries to guess best possible value for this parameter, but in some cases custom value for this parameter might give better performance. General rule for «best value» here is: number of physical cores for 1 CPU. But please note, this parameter defines maximum number, and actual number of threads launched for any individual op might (and probably will) be lower then this value.

I.e. if you have dual-cpu 8-core system with HyperThreading enabled, total number of cores reported in system will be 32, and best value for OMP_NUM_THREADS will be 8.
If you have quad-cpu 10 core system with HyperThreading enabled, total number of cores reported in system will be 80, and best value for OMP_NUM_THREADS will be 10.
If you have single cpu with 16 cores and HyperThreading enabled, total number of cores reported in system will be 32, and best value for OMP_NUM_THREADS will be somewhere between 8 and 16, depending on workload size.

## Intel MKL

By default, OpenBLAS is used with ND4j, but there’s an option to use first-class high-performance Intel MKL library together with ND4j/DL4j. That happens automatically, if you have MKL libraries on $PATH, they will be used for BLAS operations. Intel MKL is available for Linux, MacOS, Windows. It’s also possible to compile ND4j with Intel MKL pre-linked, if you have Intel MKL installed.

## Spark environment

For distributed environment, it might be good idea for some customizations. First of all, you should consider configuring number of executors per node to the value that excludes HyperThreading cores, additionally that will save you some space for internal multithreading in native operations. And another important thing, is OMP_NUM_THREADS value in such environments. Since you’re going to have high number of concurrent java threads provided by spark, native parallelism should be reduced to something like 2-4 threads. So, OMP_NUM_THREADS=4 might be good value to start perf tuning with. 

## Building from source

Please note: manual compilation requires some skills & understanding in C/C++ field. But when you’re building things from source, and you know your environment, you have few additional options that provide better performance:

Vectorized math libraries - using those, you’ll enable better use of CPU SIMD abilities for math functions, widely used in machine learning.
libsvml available with Intel MKL
libmvec available on Linux with glibc 2.22+
-march=native
This is generic compilation optimization, which enables code compilation for current hardware architecture. On modern processors that usually gives better vectorization.

## Troubleshooting CPU backend:

### ND4J_SKIP_BLAS_THREADS

If you have unusual BLAS environment, or you have troubles around Nd4jBlas.setMaxThreads() call - set environment variable  `ND4J_SKIP_BLAS_THREADS` to any value. In this case, method won’t be triggered. However, in this case you’ll also have to set `OMP_NUM_THREADS` variable manually as well.
