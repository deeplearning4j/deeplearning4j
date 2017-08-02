---
title: Deeplearning4j Benchmarks
layout: default
---

# How to Run Deeplearning4j Benchmarks

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
	                INDArray output = model.output(ds.getFeatureMatrix(), false);
	                eval.eval(ds.getLabels(), output);
            }

You can optimize by using an asychronous loader in the background. Java can do real multi-threading. It can load data in the background while other threads take care of compute. So you load data into the GPU at the same time that compute is being run. The neural net trains even as you grab new data from memory.

This is the [relevant code](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-scaleout/deeplearning4j-scaleout-parallelwrapper/src/main/java/org/deeplearning4j/parallelism/ParallelWrapper.java#L136), in particular the third line:

        MultiDataSetIterator iterator;
        if (prefetchSize > 0 && source.asyncSupported()) {
            iterator = new AsyncMultiDataSetIterator(source, prefetchSize);
        } else iterator = source;

There are actually two types of asynchronous dataset iterators. The `AsyncDataSetIterator` is what you would use most of the time. It's described in the [Javadoc here](https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/AsyncDataSetIterator.html).

For special cases such as recurrent neural nets applied to time series, or for computation graphs, you would use a `AsyncMultiDataSetIterator`, described in the [Javadoc here](https://deeplearning4j.org/doc/org/deeplearning4j/datasets/iterator/AsyncMultiDataSetIterator.html).

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
