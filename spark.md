---
title: Deeplearning4j on Spark
layout: default
---

# Deeplearning4j on Spark

Deep learning is computationally intensive, so on very large datasets, speed matters. You can tackle the problem with faster hardware (usually GPUs), optimized code and some form of parallelism. 

Data parallelism shards large datasets and hands those pieces to separate neural networks, say, each on its own core. Deeplearning4j relies on Spark and Hadoop for MapReduce, trains models in parallel and [iteratively averages](../iterativereduce.html) the parameters they produce in a central model. (Model parallelism, [discussed here by Jeff Dean et al](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf), allows models to specialize on separate patches of a large dataset without averaging.)

With Spark standalone, Deeplearning4j can run multi-threaded on your local machine; i.e. you don't need a cluster or the cloud. If you don't have Spark, please see our [Spark installation page](../sparkinstall.html).

## <a name="build">Build the Examples</a>

First `git clone` the Deeplearning4j Spark ML examples repo from Github and `cd` in:

       git clone https://github.com/deeplearning4j/scene-classification-spark
       cd scene-classification-spark

Compile the project with Maven using whichever Spark and Hadoop versions you need. 

       mvn clean package -Dspark.version=1.4.1 -Dhadoop.version=2.4.0

## <a name="run">Run the Examples</a>

Make sure you're in the `scene-classification-spark` directory. 

While the [scene classification example](https://github.com/deeplearning4j/scene-classification-spark/blob/master/src/main/java/org/deeplearning4j/SparkMnist.java) currently points to an S3 file, you will need to download [this SVM file with MNIST data](https://raw.githubusercontent.com/deeplearning4j/Canova/master/canova-api/src/test/resources/mnist_svmlight.txt) locally and point to it instead. 

Once you've done that, run a `spark-submit` command similar to this.

        bin/spark-submit --master spark://ec2-$ADDRESS_HERE.us-west-1.compute.amazonaws.com:7077 --driver-memory 3g --driver-cores 4 --executor-cores 30 --num-executors 200 --class org.deeplearning4j.SparkMnist scene-classification-spark-1.0-SNAPSHOT.jar

## OpenBLAS With Spark

To get OpenBLAS support in a Spark environment, Netlib should be in the Spark classpath. Add the application fat JAR to the Spark classpath (via `--driver-class-path` for local runs and with `spark.executor.extraClassPath` for clusters).

## Scala Resources

* [Deeplearning4j Spark Examples Repo](https://github.com/deeplearning4j/scene-classification-spark)
* [Accelerate Spark with OpenBlas](../spark-fast-native-binaries.html)
* ND4S: [N-Dimensional Arrays for Scala](https://github.com/deeplearning4j/nd4s)
* [ND4J/ND4S Benchmarks](http://nd4j.org/benchmarking)
* [ND4J, Scala & Scientific Computing](http://nd4j.org/scala.html)
* [Intro to Iterative Reduce](../iterativereduce)
