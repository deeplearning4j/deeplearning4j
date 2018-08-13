---
title: Benchmarking with DL4J and ND4J
short_title: Benchmark Guide
description: General guidelines for benchmarking in DL4J and ND4J.
category: Get Started
weight: 10
---

## General Benchmarking Guidelines

**Guideline 1: Run Warm-Up Iterations Before Benchmarking**

A warm-up period is where you run a number of iterations (for example, a few hundred) of your benchmark without timing, before commencing timing for further iterations.

Why is a warm-up required? The first few iterations of any ND4J/DL4J execution may be slower than those that come later, for a number of reasons:
1. In the initial benchmark iterations, the JVM has not yet had time to perform just-in-time compilation of code. Once JIT has completed, code is likely to execute faster for all subsequent operations
2. ND4J and DL4J (and, some other libraries) have some degree of lazy initialization: the first operation may trigger some one-off execution code.
3. DL4J or ND4J (when using workspaces) can take some iterations to learn memory requirements for execution. During this learning phase, performance will be lower than after its completion.


**Guideline 2: Run Multiple Iterations of All Benchmarks**

Your benchmark isn't the only thing running on your computer (not to mention if you are using cloud harware, that might have shared resources). And operation runtime is not perfectly deterministic.

For benchmark results to be reliable, it is important to run multiple iterations - and ideally report both mean and standard deviation for the runtime. Without this, it's impossible to compare the performance of operations, as performance differences may simply be due to random variation.



**Guideline 3: Pay Careful Attention to What You Are Benchmarking**

This is especially important when comparing frameworks. Before you declare that "performance on operation X is Y" or "A is faster than B", make sure that:

1. You are benchmarking only the operations of interest.

If your goal is to check the performance of an operation, make sure that only this operation is being timed.

You should carefully check whether you unintentionally including other things - for example, does it include:
JVM initialization time? Library initialization time? Result array allocation time? Garbage collection time? Data loading time?

Ideally, these should be excluded from any timing/performance results you report. If they cannot be excluded, make sure you note this whenever making performance claims.


2. What native libraries are you using?

For example: what BLAS implementation (MKL, OpenBLAS, etc)? If you are using CUDA, are you using CuDNN?
ND4J and DL4J can use these libraries (MKL, CuDNN) when they are available - but are not always available by default. If they are not made available, performance can be lower - sometimes considerably.

This is especially important when comparing results between libraries: for example, if you compared two libraries (one using OpenBLAS, another using MLK) your results may simply reflect the performance differences it the BLAS library being used - and not the performance oth the libraries being tested. Similarly, one library with CuDNN and another without CuDNN may simply reflect the performance benefit of using CuDNN.


3. How are things configured?

For better or worse, DL4J and ND4J allow a lot of configuration. The default values for a lot of this configuration is adequate for most users - but sometimes manual configuration is required for optimal performance. This can be especially true in some benchmarks!
Some of these configuration options allow users to trade off higher memory use for better performance, for example. Some configuration options of note:
(a) [Memory configuration](./deeplearning4j-config-memory)
(b) [Workspaces and garbage collection](./deeplearning4j-config-workspaces)
(c) [CuDNN](./deeplearning4j-config-cudnn)
(d) DL4J Cache Mode (enable using ```.cacheMode(CacheMode.DEVICE)```)


If you aren't sure if you are only measuring what you intend to measure when running DL4J or ND4J code, you can use a profiler such as VisualVM or YourKit Profilers.


4. What versions are you using?

When benchmarking, you should use the latest version of whatever libraries you are benchmarking. There's no point identifying and reporting a bottleneck that was fixed 6 months ago. An exception to this would be when you are comparing performance over time between versions.
Note also that snapshot versions of DL4J and ND4J are also available - these may contain performance improvements (feel free to ask)


**Guideline 4: Focus on Real-World Use Cases - And Run a Range of Sizes**

Consider for example a benchmark a benchmark that adds two numbers:
```
double x = 0;
//<start timing>
x += 1.0;
//<end timing>
```

And something equivalent in ND4J:
```
INDArray x = Nd4j.create(1);
//<start timing>
x.addi(1.0);
//<end timing>
```

Of course, the ND4J benchmark above is going to be much slower - method calls are required, input validation is performed, native code has to be called (with context switching overhead), and so on. One must ask the question, however: is this what users will actually be doing with ND4J or an equivalent linear algebra library? It's an extreme example - but the general point is a valid one.


Note also that performance on mathematical operations can be size - and shape - specific.
For example, if you are benchmarking the performance on matrix multiplication - the matrix dimensions can matter a lot. In some internal benchmarks, we found that different BLAS implementations (MKL vs OpenBLAS) - and different backends (CPU vs GPU) - can perform very differently with different matrix dimensions. None of the BLAS implementations (OpenBLAS, MKL, CUDA) we have tested internally were uniformly faster than others for all input shapes and sizes.

Therefore - whenever you are running benchmarks, it's important to run those benchmarks with multiple different input shapes/sizes, to get the full performance picture.


**Guideline 5: Understand Your Hardware**

When comparing different hardware, it's important to be aware of what it excels at.
For example, you might find that neural network training performs faster on a CPU with minibatch size 1 than on a GPU - yet larger minibatch sizes show exactly the opposite. Similarly, small layer sizes may not be able to adequately utilize the power of a GPU.

Furthermore, some deep learning distributions may need to be specifically compiled to provide support for hardware features such as AVX2 (note that recent version of ND4J are packaged with binaries for CPUs that support these features). When running benchmarks, the utilization (or lack there-of) of these features can make a considerable difference to performance.


**Guideline 6: Make It Reproducible**

When running benchmarks, it's important to make your benchmarks reproducible.
Why? Good or bad performance may only occur under certain limited circumstances.

And finally - remember that (a) ND4J and DL4J are in constant development, and (b) benchmarks do sometimes identify performance bottlenecks (after all we - ND4J includes literally hundreds of distinct operations). If you identify a performance bottleneck, great - we want to know about it - so we can fix it. Any time a potential bottleneck is identified, we first need to reproduce it - so that we can study it, understand it and ultimately fix it.

**Guideline 7: Understand the Limitations of Your Benchmarks**

Linear algebra libraries contain hundreds of distinct operations. Neural network libraries contain dozens of layer types. When benchmarking, it's important to understand the limitations of those benchmarks. Benchmarking one type of operation or layer cannot tell you anything about the performance on other types of layers or operations - unless they share code that has been identified to be a performance bottleneck.

**Guideline 8: If You Aren't Sure - Ask**

The DL4J/ND4J developers are available on Gitter. You can ask questions about benchmarking and performance there: [https://gitter.im/deeplearning4j/deeplearning4j](https://gitter.im/deeplearning4j/deeplearning4j)

And if you do happen to find a performance issue - let us know!



## ND4J Specific Benchmarking


**A Note on BLAS and Array Orders**

BLAS - or Basic Linear Algebra Subprograms - refers to an interface and set of methods used for linear algebra operations. Some examples include 'gemm' - General Matrix Multiplication - and 'axpy', which implements ```Y = a*X+Y```.


ND4J can use multiple BLAS implementations - versions up to and including 1.0.0-beta have defaulted to OpenBLAS. However, if Intel MKL (free versions are available [here](https://software.intel.com/en-us/mkl)) is installed an available, ND4J will link with it for improved performance in many BLAS operations.

Note that ND4J will log the BLAS backend used when it initializes. For example:
```
14:17:34,169 INFO  ~ Loaded [CpuBackend] backend
14:17:34,672 INFO  ~ Number of threads used for NativeOps: 8
14:17:34,823 INFO  ~ Number of threads used for BLAS: 8
14:17:34,831 INFO  ~ Backend used: [CPU]; OS: [Windows 10]
14:17:34,831 INFO  ~ Cores: [16]; Memory: [7.1GB];
14:17:34,831 INFO  ~ Blas vendor: [OPENBLAS]
```


Performance can depend on the available BLAS library - in internal tests, we have found that OpenBLAS has been between 30% faster and 8x slower than MKL - depending on the array sizes and array orders.

Regarding array orders, this also matters for performance. ND4J has the possibility of representing arrays in either row major ('c') or column major ('f') order. See [this Wikipedia page](https://en.wikipedia.org/wiki/Row-_and_column-major_order) for more details. Performance in operations such as matrix multiplication - but also more general ND4J operations - depends on the input and result array orders.

For matrix multiplication, this means there are 8 possible combinations of array orders (c/f for each of input 1, input 2 and result arrays). Performance won't be the same for all cases.

Similarly, an operation such as element-wise addition (i.e., z=x+y) will be much faster for some combinations of input orders than others - notably, when x, y and z are all the same order. In short, this is due to memory striding: it's cheaper to read a sequencee of memory addresses when those memory addresess are adjacent to each other in memory, as compared to being spread far apart.

Note that, by default, ND4J expects result arrays (for matrix multiplication) to be defined in column major ('f') order, to be consistent across backends, given that CuBLAS (i.e., NVIDIA's BLAS library for CUDA) requires results to be in f order. As a consequence, some ways of performing matrix multiplication with the result array being in c order will have lower performance than if the same operation was executed with an 'f' order array.

Finally, when it comes to CUDA: array orders/striding can matter even more than when running on CPU. For example, certain combinations of orders can be much faster than others - and input/output dimesions that are even multiples of 32 or 64 typically perform faster (sometimes considerably) than when input/output dimensions are not multiples of 32.



## DL4J Specific Benchmarking


Most of what has been said for ND4J also applies to DL4J.

In addition:
1. If you are using the nd4j-native (CPU) backend, ensure you are using Intel MKL. This is faster than the default of OpenBLAS in most cases.
2. If you are using CUDA, ensure you are using CuDNN ([link](./deeplearning4j-config-cudnn)
3. Check the [Workspaces](./deeplearning4j-config-workspaces) and [Memory](./deeplearning4j-config-memory) guides. The defaults are usually good - but sometimes better performance can be obtained with some tweaking. This is especially important if you have a lot of Java objects (such as, Word2Vec vectors) in memory while training.
4. Watch out for ETL bottlenecks. You can add PerformanceListener to your network training to see if ETL is a bottleneck.
5. Don't forget that performance is dependent on minibatch sizes. Don't benchmark with minibatch size 1 - use something more realistic.
6. If you need multi-GPU training or inference support, use ParallelWrapper or ParallelInference.
7. Don't forget that CuDNN is configurable: you can specify DL4J/CuDNN to prefer performance - at the expense of memory - using ```.cudnnAlgoMode(ConvolutionLayer.AlgoMode.PREFER_FASTEST)``` configuration on convolution layers
8. When using GPUs, multiples of 8 (or 32) for input sizes and layer sizes may perform better.
9. When using RNNs (and manually creating INDArrays), use 'f' ordered arrays for both features and (RnnOutputLayer) labels. Otherwise, use 'c' ordered arrays. This is for faster memory access.


## Common Benchmark Mistakes

Finally, here's a summary list of common benchmark mistakes:

1. Not using the latest version of ND4J/DL4J (there's no point identifying a bottleneck that was fixed many releases back). Consider trying snapshots to get the latest performance improvements.
2. Not paying attention to whan native libraries (MKL, OpenBLAS, CuDNN etc) are being used
3. Providing no warm-up period before benchmarking begins
4. Running only a single (or too few) iterations, or not reporting mean, standard deviation and number of iterations
5. Not configuring workspaces, garbage collection, etc
6. Running only one possible case - for example, benchmarking a single set of array dimensions/orders when benchmarking BLAS operations
7. Running unusually small inputs - for example, minibatch size 1 on a GPU (which might be slower - but isn't realistic!)
8. Not measuring exactly - and only - what you claim to be measuring (for example, not accounting for array allocation, initialization or garbage collection time)
9. Not making your benchmarks reprodicable (does the benchmark conclusion generalize? are there problems with the benchmark? what can we do to fix it?)
10. Comparing results across different hardware, not accounting for differences (for example, testing on one machine with AVX2 support, and on another without)
11. Not asking the devs (via the [DL4J/ND4J Gitter Channel](https://gitter.im/deeplearning4j/deeplearning4j) - we are happy to provide suggestions and investigate if performance isn't where it should be!






# How to Run Deeplearning4j Benchmarks - A Guide

Total training time is always ETL plus computation. That is, both the data pipeline and the matrix manipulations determine how long a neural network takes to train on a dataset. 

When programmers familiar with Python try to run benchmarks comparing Deeplearning4j to well-known Python frameworks, they usually end up comparing ETL + computation on DL4J to just computation on the Python framework. That is, they're comparing apples to oranges. We'll explain how to optimize several parameters below. 

The JVM has knobs to tune, and if you know how to tune them, you can make it a very fast environment for deep learning. There are several things to keep in mind on the JVM. You need to:

* Increase the [heap space](http://javarevisited.blogspot.com/2011/05/java-heap-space-memory-size-jvm.html)
* Get garbage collection right
* Make ETL asynchronous
* Presave datasets (aka pickling)

## Setting Heap Space

Users have to reconfigure their JVMs themselves, including setting the heap space. We can't give it to you preconfigured, but we can show you how to do it. Here are the two most important knobs for heap space.

* Xms sets the minimum heap space
* Xmx sets the maximum heap space

You can set these in IDEs like IntelliJ and Eclipse, as well as via the CLI like so:

		java -Xms256m -Xmx1024m YourClassNameHere

In [IntelliJ, this is a VM parameter](https://www.jetbrains.com/help/idea/2016.3/setting-configuration-options.html), not a program argument. When you hit run in IntelliJ (the green button), that sets up a run-time configuration. IJ starts a Java VM for you with the configurations you specify. 

What’s the ideal amount to set `Xmx` to? That depends on how much RAM is on your computer. In general, allocate as much heap space as you think the JVM will need to get work done. Let’s say you’re on a 16G RAM laptop — allocate 8G of RAM to the JVM. A sound minimum on laptops with less RAM would be 3g, so 

		java -Xmx3g

It may seem counterintuitive, but you want the min and max to be the same; i.e. `Xms` should equal `Xmx`. If they are unequal, the JVM will progressively allocate more memory as needed until it reaches the max, and that process of gradual allocation slows things down. You want to pre-allocate it at the beginning. So 

		java -Xms3g -Xmx3g YourClassNameHere

IntelliJ will automatically specify the [Java main class](https://docs.oracle.com/javase/tutorial/getStarted/application/) in question.

Another way to do this is by setting your environmental variables. Here, you would alter your hidden `.bash_profile` file, which adds environmental variables to bash. To see those variables, enter `env` in the command line. To add more heap space, enter this command in your console:

		echo "export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=512m"" > ~/.bash_profile

We need to increase heap space because Deeplearning4j loads data in the background, which means we're taking more RAM in memory. By allowing more heap space for the JVM, we can cache more data in memory. 

## Garbage Collection

A garbage collector is a program which runs on the JVM and gets rid of objects no longer used by a Java application. It is automatic memory management. Creating a new object in Java takes on-heap memory: A new Java object takes up 8 bytes of memory by default. So every new `DatasetIterator` you create takes another 8 bytes. 

You may need to alter the garbage collection algorithm that Java is using. This can be done via the command line like so:

		java -XX:+UseG1GC

Better garbage collection increases throughput. For a more detailed exploration of the issue, please read this [InfoQ article](https://www.infoq.com/articles/Make-G1-Default-Garbage-Collector-in-Java-9).

DL4J is tightly linked to the garbage collector. [JavaCPP](https://github.com/bytedeco/javacpp), the bridge between the JVM and C++, adheres to the heap space you set with `Xmx` and works extensively with off-heap memory. The off-heap memory will not surpass the amount of heap space you specify. 

JavaCPP, created by a Skymind engineer, relies on the garbage collector to tell it what has been done. We rely on the Java GC to tell us what to collect; the Java GC points at things, and we know how to de-allocate them with JavaCPP. This applies equally to how we work with GPUs. 

The larger the batch size you use, the more RAM you’re taking in memory. 

## ETL & Asynchronous ETL

In our `dl4j-examples` repo, we don't make the ETL asynchronous, because the point of examples is to keep them simple. But for real-world problems, you need asynchronous ETL, and we'll show you how to do it with examples. 

Data is stored on disk and disk is slow. That’s the default. So you run into bottlenecks when loading data onto your harddrive. When optimizing throughput, the slowest component is always the bottleneck. For example, a distributed Spark job using three GPU workers and one CPU worker will have a bottleneck with the CPU. The GPUs have to wait for that CPU to finish. 

The Deeplearning4j class `DatasetIterator` hides the complexity of loading data on disk. The code for using any Datasetiterator will always be the same, invoking looks the same, but they work differently. 

* one loads from disk 
* one loads asynchronously
* one loads pre-saved from RAM

Here's how the DatasetIterator is uniformly invoked for MNIST:

            while(mnistTest.hasNext()){
	                DataSet ds = mnistTest.next();
	                INDArray output = model.output(ds.getFeatures(), false);
	                eval.eval(ds.getLabels(), output);
            }

You can optimize by using an asychronous loader in the background. Java can do real multi-threading. It can load data in the background while other threads take care of compute. So you load data into the GPU at the same time that compute is being run. The neural net trains even as you grab new data from memory.

This is the [relevant code](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-scaleout/deeplearning4j-scaleout-parallelwrapper/src/main/java/org/deeplearning4j/parallelism/ParallelWrapper.java#L136), in particular the third line:

        MultiDataSetIterator iterator;
        if (prefetchSize > 0 && source.asyncSupported()) {
            iterator = new AsyncMultiDataSetIterator(source, prefetchSize);
        } else iterator = source;

There are actually two types of asynchronous dataset iterators. The `AsyncDataSetIterator` is what you would use most of the time. It's described in the [Javadoc here](https://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/datasets/iterator/AsyncDataSetIterator.html).

For special cases such as recurrent neural nets applied to time series, or for computation graphs, you would use a `AsyncMultiDataSetIterator`, described in the [Javadoc here](https://deeplearning4j.org/api/{{page.version}}/org/deeplearning4j/datasets/iterator/AsyncMultiDataSetIterator.html).

Notice in the code above that `prefetchSize` is another parameter to set. Normal batch size might be 1000 examples, but if you set `prefetchSize` to 3, it would pre-fetch 3,000 instances.

## ETL: Comparing Python frameworks With Deeplearning4j

In Python, programmers are converting their data into [pickles](https://docs.python.org/2/library/pickle.html), or binary data objects. And if they're working with a smallish toy dataset, they're loading all those pickles into RAM. So they're effectively sidestepping a major task in dealing with larger datasets. At the same time, when benchmarking against Dl4j, they're not loading all the data onto RAM. So they're effectively comparing Dl4j speed for training computations + ETL against only training computation time for Python frameworks. 

But Java has robust tools for moving big data, and if compared correctly, is much faster than Python. The Deeplearning4j community has reported up to 3700% increases in speed over Python frameworks, when ETL and computation are optimized.

Deeplearning4j uses DataVec as it ETL and vectorization library. Unlike other deep-learning tools, DataVec does not force a particular format on your dataset. (Caffe forces you to use [hdf5](https://support.hdfgroup.org/HDF5/), for example.)

We try to be more flexible. That means you can point DL4J at raw photos, and it will load the image, run the transforms and put it into an NDArray to generate a dataset on the fly. 

But if your training pipeline is doing that every time, Deeplearning4j will seem about 10x slower than other frameworks, because you’re spending your time creating datasets. Every time you call `fit`, you're recreating a dataset, over and over again. We allow it to happen for ease of use, but we can show you how to speed things up. There are ways to make it just as fast. 

One way is to pre-save the datasets, in a manner similar to the Python frameworks. (Pickles are pre-formatted data.) When you pre-save the dataset, you create a separate class.

Here’s how you [pre-save datasets](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/presave/PreSave.java).

A `Recordreaderdatasetiterator` talks to Datavec and outputs datasets for DL4J. 

Here’s how you [load a pre-saved dataset](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/misc/presave/LoadPreSavedLenetMnistExample.java).

Line 90 is where you see the asynchronous ETL. In this case, it's wrapping the pre-saved iterator, so you're taking advantage of both methods, with the asynch loading the pre-saved data in the background as the net trains. 

## MKL and Inference on CPUs

If you are running inference benchmarks on CPUs, make sure you are using Deeplearning4j with Intel's MKL library, which is available via a clickwrap; i.e. Deeplearning4j does not bundle MKL like Anaconda, which is used by libraries like PyTorch. 