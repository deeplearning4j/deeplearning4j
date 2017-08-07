---
title: Deeplearning4j Benchmarks
layout: cn-default
---

# Deeplearning4j Benchmarks

Total training time is always ETL plus computation. That is, both the data pipeline and the matrix manipulations determine how long a neural network takes to train on a dataset.

When programmers familiar with Python try to run benchmarks comparing Deeplearning4j to other frameworks, they usually end up comparing ETL + computation on DL4J to computation on a Python framework. That is, they're comparing apples to oranges.

The JVM has knobs to tune, and if you know how to tune them, you can make it a very fast environment for deep learning. There are three things to keep in mind on the JVM:

* You need to increase the [heap space](http://javarevisited.blogspot.com/2011/05/java-heap-space-memory-size-jvm.html)
* You need to get garbage collection right
* You need asynchronous ETL

## Heap Space

Users have to reconfigure their JVMs themselves re:heap space. We can't give it to you preconfigured, but we can show you how to do it.

* To increase heap space, you'll need to find and alter your hidden `.bash_profile` file, which adds environmental variables to bash. To see those variables, enter `env` in the command line. To add more heap space, enter this command in your console:
		echo "export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=512m"" > ~/.bash_profile

## ETL

On our examples, we don't make the ETL asynchronous, because the point of examples is to keep them simple. But for real world problems, you need to do that. Asynchronous ETL means TK...

## Python

In Python, programmers are converting their data into pickles, or binary data objects. And if they're working with a smallish toy dataset, they're loading all those pickles into RAM. So they're effectively sidestepping a major task in dealing with larger datasets. At the same time, when benchmarking against Dl4j, they're not loading all the data onto RAM. So they're effectively comparing Dl4j speed for training computations + ETL against only training computation time for Python frameworks.

But Java has robust tools for moving big data, and if compared correctly, is much faster than Python. The Deeplearning4j community has reported up to 3700% increases in speed over Python frameworks, when ETL and computation are optimized.
