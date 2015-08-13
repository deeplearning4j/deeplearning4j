---
title: 
layout: default
---

# Architecture

DeepLearning4j uses a handful of technologies for normal and distributed training, as well as prediction with neural networks. Below you'll find the current architecture, an alpha version that is likely to change with the welcome feedback of our users.

### Matrix computations

[Jblas](http://mikiobraun.github.io/jblas/) uses a Fortran library for matrix computations. We chose to use Jblas, a native dependency, because its matrix computations are faster than other options, and deep learning is nothing if not a series of matrix computations. The work of deep-learning nets is too computationally intense to select slow tools and expect timely results. 

### distributed service discovery

[Zookeeper](http://zookeeper.apache.org/) does distributed service discovery and configuration/cluster coordination well. It comes embedded in many environments, and powers a number of well-known technologies, like Storm and Hadoop.

### Clustering / distributed computing

[Akka](http://akka.io/) is a parallel framework written in Scala that powers [Spark](http://spark.apache.org). It has a great programming paradigm and transparent parallel computation via the actor model. This allows for training at scale as well as very simple, serializable models at prediction time.

### Distributed data structures

[Hazelcast](http://hazelcast.org/) is a distributed data structure framework for handling common object serialization requirements. It's primarily used to create transparent data structures for state tracking. While Hazelcast may seem redundant alongside zookeeper, zookeeper does not encourage and should not be used for distributed serialization stores.
