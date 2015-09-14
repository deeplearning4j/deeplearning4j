---
layout: default
---

# Deeplearning4j on Spark

Given that deep learning is computationally intensive, if you're working with large datasets, you should think about how to train deep neural networks in parallel. 

With Spark standalone, Deeplearning4j can run multi-threaded on your local machine; i.e. you don't need a cluster or the cloud. If you don't have Spark, please see our [Spark installation page](../sparkinstall.html).

## Build the Examples

First `git clone` the Deeplearning4j Spark ML examples repo from Github and `cd` in:

       git clone https://github.com/deeplearning4j/dl4j-spark-ml-examples
       cd dl4j-spark-ml-examples

Compile the project with Maven using whichever Spark and Hadoop versions you need. 

       mvn clean package -Dspark.version=1.4.1 -Dhadoop.version=2.4.0

## Run the Examples

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

## Just Give Me the Code

Here's an iPython Notebook for Iris classification using [Deeplearning4j and Spark](https://github.com/deeplearning4j/dl4j-spark-ml-examples/blob/master/notebooks/dl4j-iris.ipynb). 
