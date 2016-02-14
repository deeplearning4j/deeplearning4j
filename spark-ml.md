---
layout: default
title: Deeplearning4j on Spark
---

# Deeplearning4j on Spark

Content

* [Build with the CLI](#build)
* [Run with the CLI](#run)
* [Spark Notebook](#notebook)

Deep learning is computationally intensive, so on very large datasets, speed matters. You can tackle the problem with faster hardware (usually GPUs), optimized code and some form of parallelism. 

Data parallelism shards large datasets and hands those pieces to separate neural networks, say, each on its own core. Deeplearning4j relies on Spark and Hadoop for MapReduce, trains models in parallel and [iteratively averages](../iterativereduce.html) the parameters they produce in a central model. (Model parallelism, [discussed here by Jeff Dean et al](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf), allows models to specialize on separate patches of a large dataset without averaging.)

With Spark standalone, Deeplearning4j can run multi-threaded on your local machine; i.e. you don't need a cluster or the cloud. If you don't have Spark, please see our [Spark installation page](../sparkinstall.html).

## <a name="build">Build the Examples</a>

First `git clone` the Deeplearning4j Spark ML examples repo from Github and `cd` in:

       git clone https://github.com/deeplearning4j/dl4j-spark-ml-examples
       cd dl4j-spark-ml-examples

Compile the project with Maven using whichever Spark and Hadoop versions you need. 

       mvn clean package -Dspark.version=1.4.1 -Dhadoop.version=2.4.0

## <a name="run">Run the Examples</a>

Then make sure you're in the dl4j-spark-ml-examples directory and run

        bin/run-example ml.JavaIrisClassification

The output, amid a river of other log info, should look like this:

![Alt text](../img/dl4j_iris_dataframe.png)

You can run other examples with the `bin/run-example` command:

    bin/run-example
    Usage: ./bin/run-example <example-class> [example-args]
        - set env variable MASTER to use a specific master; e.g. export MASTER=local[*]
        - can use abbreviated example class name relative to org.deeplearning4j
        (e.g. ml.JavaIrisClassification,ml.JavaLfwClassification)

They can be run on your local machine by setting `master` to `local[YourNumberOfCores]` or `local[*]` for all cores. For example: 

    MASTER=local[*] bin/run-example ml.JavaIrisClassification 
    //or
    export MASTER=local[*]
    bin/run-example ml.JavaIrisClassification

You can read more about Spark standalone, which is just Spark without a cluster, [here](http://spark.apache.org/docs/latest/spark-standalone.html). Ultimately, training neural networks with Deeplearning4j is just another Spark job. 

## <a name="notebook">Spark Notebook for Iris Dataset</a>

Here's a Spark Notebook for Iris classification using [Deeplearning4j and Spark](https://github.com/deeplearning4j/dl4j-spark-ml-examples/blob/master/notebooks/dl4j-iris.ipynb). 

## Scala Resources

* [Deeplearning4j-Spark-ML Repo](https://github.com/deeplearning4j/dl4j-spark-ml)
* ND4S: [N-Dimensional Arrays for Scala](https://github.com/deeplearning4j/nd4s)
* [ND4J/ND4S Benchmarks](http://nd4j.org/benchmarking)
* [ND4J, Scala & Scientific Computing](http://nd4j.org/scala.html)
* [Intro to Iterative Reduce](../iterativereduce)
