---
title: Deeplearning4j Benchmarks
layout: default
---

# Deeplearning4j Benchmarks

Total training time is always ETL plus computation. That is, both the data pipeline and the matrix manipulations determine how long a neural network takes to train on a dataset. 

When programmers familiar with Python try to run benchmarks comparing Deeplearning4j to well-known Python frameworks, they usually end up comparing ETL + computation on DL4J to just computation on the Python framework. That is, they're comparing apples to oranges. We'll explain how to optimize several parameters below. 

The JVM has knobs to tune, and if you know how to tune them, you can make it a very fast environment for deep learning. There are several things to keep in mind on the JVM. You need to:

* Increase the [heap space](http://javarevisited.blogspot.com/2011/05/java-heap-space-memory-size-jvm.html)
* Get garbage collection right
* Make ETL asynchronous
* Presave datasets (aka pickling)

## Heap Space

Users have to reconfigure their JVMs themselves, including setting the heap space. We can't give it to you preconfigured, but we can show you how to do it. Here are the two most important knobs for heap space.

* Xms sets the minimum heap space
* Xmx sets the maximum heap space

You can set these in IDEs like IntelliJ and Eclipse, as well as via the CLI like so:

		java -Xms256m -Xmx1024m YourClassNameHere

In [IntelliJ, this is a VM parameter](https://www.jetbrains.com/help/idea/2016.3/setting-configuration-options.html), not a program argument. When you hit run in IntelliJ (the green button), that sets up a run-time configuration. IJ starts a Java VM for you with the configurations you specify. 

What’s the ideal amount to set `Xmx` to? That depends on how much RAM is on your computer. In general, allocate as much heap space as you think the JVM will need to get work done. Let’s say you’re on a 16G RAM laptop — allocate 8G of RAM to the JVM. A sound minimum on laptops with less RAM would be 3g, so 

		java -Xmx3g

It may seem coutnerintuitive, but you want the min and max to be the same; i.e. `Xms` should equal `Xmx`. If they are unequal, the JVM will progressively allocate more memory as needed until it reaches the max, and that process of gradual allocation slows things down. You want to pre-allocate it at the beginning. So 

		java -Xms3g -Xmx3g YourClassNameHere

IntelliJ will automatically specify the [Java main class](https://docs.oracle.com/javase/tutorial/getStarted/application/) in question.

* To increase heap space, you'll need to find and alter your hidden `.bash_profile` file, which adds environmental variables to bash. To see those variables, enter `env` in the command line. To add more heap space, enter this command in your console:

		echo "export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=512m"" > ~/.bash_profile



## Garbage Collection

## ETL & Asynchronous ETL

In our dl4j-examples repo, we don't make the ETL asynchronous, because the point of examples is to keep them simple. But for real-world problems, you need to do so.

## ETL: Comparing Python frameworks With Deeplearning4j

In Python, programmers are converting their data into [pickles](https://docs.python.org/2/library/pickle.html), or binary data objects. And if they're working with a smallish toy dataset, they're loading all those pickles into RAM. So they're effectively sidestepping a major task in dealing with larger datasets. At the same time, when benchmarking against Dl4j, they're not loading all the data onto RAM. So they're effectively comparing Dl4j speed for training computations + ETL against only training computation time for Python frameworks. 

But Java has robust tools for moving big data, and if compared correctly, is much faster than Python. The Deeplearning4j community has reported up to 3700% increases in speed over Python frameworks, when ETL and computation are optimized.
