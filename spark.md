---
title: Deeplearning4j on Spark
layout: default
---

# Deeplearning4j on Spark

Deep learning is computationally intensive, so on very large datasets, speed matters. You can tackle the problem with faster hardware (usually GPUs), optimized code and some form of parallelism. 

Data parallelism shards large datasets and hands those pieces to separate neural networks, say, each on its own core. Deeplearning4j relies on Spark and Hadoop for MapReduce, trains models in parallel and [iteratively averages](../iterativereduce.html) the parameters they produce in a central model. (Model parallelism, [discussed here by Jeff Dean et al](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf), allows models to specialize on separate patches of a large dataset without averaging.)

With Spark standalone, Deeplearning4j can run multi-threaded on your local machine; i.e. you don't need a cluster or the cloud. If you don't have Spark, please see our [Spark installation page](../sparkinstall.html).

## Spark Examples

DL4J's Spark examples repository is here:
[https://github.com/deeplearning4j/dl4j-spark-cdh5-examples](https://github.com/deeplearning4j/dl4j-spark-cdh5-examples)

The examples at the above repository are set up to run using Spark local, however they can be adapted to run on a cluster by (a) removing the setMaster() configuration option, and (b) running them using Spark submit.



## OpenBLAS With Spark

To get OpenBLAS support in a Spark environment, Netlib should be in the Spark classpath. Add the application fat JAR to the Spark classpath (via `--driver-class-path` for local runs and with `spark.executor.extraClassPath` for clusters).

## Scala Resources

* [Deeplearning4j Spark Examples Repo](https://github.com/deeplearning4j/dl4j-spark-cdh5-examples)
* [Accelerate Spark with OpenBlas](../spark-fast-native-binaries.html)
* ND4S: [N-Dimensional Arrays for Scala](https://github.com/deeplearning4j/nd4s)
* [ND4J/ND4S Benchmarks](http://nd4j.org/benchmarking)
* [ND4J, Scala & Scientific Computing](http://nd4j.org/scala.html)
* [Intro to Iterative Reduce](../iterativereduce)
