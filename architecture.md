---
title: 
layout: default
---

# architecture

DeepLearning4j uses a host of technologies for normal and distributed training, and prediction of neural networks.

Below is the current architecture, an alpha version that is likely to change, based on the welcome feedback of our users.


### matrix computations

[Jblas](http://mikiobraun.github.io/jblas/) uses Fortran library for matrix computations. We chose to use Jblas as a native dependency because its matrix computations are faster than other options, and deep learning is nothing if not a series of matrix computations. The work of deep-learning nets is too computationally intense to select slow tools and expect timely results. 

### distributed service discovery

[Zookeeper](http://zookeeper.apache.org/) does distributed service discovery and configuration/cluster coordination well. It comes embedded in many environments, and powers a number of well-known technologies, like Storm and Hadoop.

### clustering / distributed computing

[Akka](http://akka.io/) is the great parallel framework written in Scala that powers [Spark](http://spark.apache.org). It has a great programming paradigm and transparent parallel computation via the actor model. This allows for training at scale as well as very simple, serializable models at prediction time.

### distributed data structures

[Hazelcast](http://hazelcast.org/)

Hazelcast is a great distributed data structure framework for handling

common object serialization requirements. Hazelcast is primarily being used for transparent data structures for state tracking.

This may seem redundant with zookeeper, but zookeeper does not encourage/should not be used for distributed serialization stores.

### optimization algorithms

[Mallet](http://mallet.cs.umass.edu/optimization.php) is used for the optimization algorithms.

The default optimization algorithm used is conjugate gradient. There is also an implementation of

gradient descent in here as well. Descent and weight decrements are implemented via a reverse 

objective function. Since mallet is a maximizer rather than a minimzer, this will achieve

the intended affect wrt the objective functions and thus allow the use of maximization search algorithms.

