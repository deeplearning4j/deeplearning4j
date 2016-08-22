---
title: Deeplearning4j on Spark
layout: default
---

# Deeplearning4j on Spark

Deep learning is computationally intensive, so on very large datasets, speed matters. You can tackle the problem with faster hardware (usually GPUs), optimized code and some form of parallelism. 

Data parallelism shards large datasets and hands those pieces to separate neural networks, say, each on its own core. Deeplearning4j relies on Spark for this, training models in parallel and [iteratively averages](./iterativereduce.html) the parameters they produce in a central model. (Model parallelism, [discussed here by Jeff Dean et al](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf), allows models to specialize on separate patches of a large dataset without averaging.)

## Overview

Deeplearning4j supports training neural networks on a Spark cluster, in order to accelerate network training.

Similar to DL4J's MultiLayerNetwork and ComputationGraph classes, DL4J defines two classes for training neural networks on Spark:

- SparkDl4jMultiLayer, a wrapper around MultiLayerNetwork
- SparkComputationGraph, a wrapper around ComputationGraph

Because these two classes are wrappers around the stardard single-machine classes, the network configuration process (i.e., creating a MultiLayerConfiguration or ComputationGraphConfiguration) is identical in both standard and distributed training. Distributed on Spark differs from local training in two respects, however: how data is loaded, and how training is set up (requiring some additional cluster-specific configuration).

The typical workflow for training a network on a Spark cluster (using spark-submit) is as follows.

1. Create your network training class. Typically, this will involve code for:
    * Specifying your network configuration (MultiLayerConfiguration or ComputationGraphConfiguration), as you would for single-machine training
    * Creating a TrainingMaster instance: this specifies how distributed training will be conducted in practice (more on this later)
    * Creating the SparkDl4jMultiLayer or SparkComputationGraph instance using the network configuration and TrainingMaster objects
    * Load your training data. There are a number of different methods of loading data, with different tradeoffs; further details will be provided in future documentation
    * Calling the appropriate ```fit``` method on the SparkDl4jMultiLayer or SparkComputationGraph instance
    * Saving or using the trained network (the trained MultiLayerNetwork or ComputationGraph instance)
2. Package your jar file ready for Spark submit
    * If you are using maven, running "mvn package -DskipTests" is one approach
3. Call Spark submit with the appropriate launch configuration for your cluster


**Note**: For single machine training, Spark local *can* be used with DL4J, though this is not recommended (due to the synchronization and serialization overheads of Spark). Instead, consider the following:

* For single CPU/GPU systems, use standard MultiLayerNetwork or ComputationGraph training
* For multi-CPU/GPU systems, use [ParallelWrapper](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j-core/src/main/java/org/deeplearning4j/parallelism/ParallelWrapper.java). This is functionally equivalent to running Spark in local mode, though has lower overhead (and hence provides better training performance). 

## How Distributed Network Training Occurs with DL4J on Spark

The current version of DL4J uses a process of parameter averaging in order to train a network. Future versions may additionally include other distributed network training approaches.


The process of training a network using parameter averaging is conceptually quite simple:

1. The master (Spark driver) starts with an initial network configuration and parameters
2. Data is split into a number of subsets, based on the configuration of the TrainingMaster
3. Iterate over the data splits. For each split of the training data:
    * Distribute the configuration, parameters (and if applicable, network updater state for momentum/rmsprop/adagrad) from the master to each worker
    * Fit each worker on its portion of the split
    * Average the parameters (and if applicable, updater state) and return the averaged results to the master
4. Training is complete, with the master having a copy of the trained network

For example, the diagram below shows the parameter averaging process with 5 workers (W1, ..., W5) and a parameter averaging frequency of 1.
Just as with offline training, a training data set is split up into a number of subsets (generally known as minibatches, in the non-distributed setting); training proceeds over each split, with each worker getting a subset of the split. In practice, the number of splits is determined automatically, based on the training configuration (based on number of workers, averaging frequency and worker minibatch sizes - see configuration section).

![Parameter Averaging](./img/parameter_averaging.png)

## A Minimal Example

This section shows the minimal set of components that you need in order to train a network on Spark.
Details on the various approaches to loading data are forthcoming.

```java
    JavaSparkContent sc = ...;
    JavaRDD<DataSet> trainingData = ...;
    MultiLayerConfiguration networkConfig = ...;

    //Create the TrainingMaster instance
    int examplesPerDataSetObject = 1;
    TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
            .(other configuration options)
            .build();

    //Create the SparkDl4jMultiLayer instance
    SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, networkConfig, trainingMaster);

    //Fit the network using the training data:
    sparkNetwork.fit(trainingData);
```

## Configuring the TrainingMaster

A TrainingMaster in DL4J is an abstraction (interface) that allows for multiple different training implementations to be used with SparkDl4jMultiLayer and SparkComputationGraph. 

Currently DL4J has one implementation, the ParameterAveragingTrainingMaster. This implements the parameter averaging process shown in the image above.
To create one, use the builder pattern:

```java
    TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(int dataSetObjectSize)
            ... (your configuration here)
            .build();
```

The ParameterAveragingTrainingMaster defines a number of configuration options that control how training is executed:

* **dataSetObjectSize**: Required option. This is specified in the builder constructor. This value specifies how many examples are in each DataSet object. As a general rule,
    * If you are training with pre-processed DataSet objects, this will be the size of those preprocessed DataSets
    * If you are training directly from Strings (for example, CSV data to a ```RDD<DataSet>``` though a number of steps) then this will usually be 1
* **batchSizePerWorker**: This controls the minibatch size for each worker. This is analagous to the minibatch size used when training on a single machine. Put another way: it is the number of examples used for each parameter update in each worker.
* **averagingFrequency**: This controls how frequently the parameters are averaged and redistributed, in terms of number of minibatches of size batchSizePerWorker. As a general rule:
    * Low averaging periods (for example, averagingFrequency=1) may be inefficient (too much network communication and initialization overhead, relative to computation)
    * Large averaging periods (for example, averagingFrequency=200) may result in poor performance (parameters in each worker instance may diverge significantly)
    * Averaging periods in the range of 5-10 minibatches is usually a safe default choice.
* **workerPrefetchNumBatches**: Spark workers are capable of asynchorously prefetching a number of minibatches (DataSet objects), to avoid waiting for the data to be loaded.
    * Setting this value to 0 disables prefetching.
    * A value of 2 is often a sensible default. Much larger values are unlikely to help in many circumstances (but will use more memory)
* **saveUpdater**: In DL4J, training methods such as momentum, RMSProp and AdaGrad are known as 'updaters'. Most of these updaters have internal history or state.
    * If saveUpdater is set to true: the updater state (at each worker) will be averaged and returned to the master along with the parameters; the current updater state will also be distributed from the master to the workers. This adds extra time and network traffic, but may improve training results.
    * If saveUpdater is set to false: the updater state (at each worker) is discarded, and the updater is reset/reinitialized in each worker.
* **repartition**: Configuration setting for when data should be repartitioned. The ParameterAveragingTrainingMaster does a mapParititons operation; consequently, the number of partitions (and, the values in each partition) matters a lot for proper cluster utilization. However, repartitioning is not a free operation, as some data necessarily has to be copied across the network. The following options are available:
    * Always: Default option. That is, repartition data to ensure the correct number of partitions
    * Never: Never repartition the data, no matter how imbalanced the partitions may be.
    * NumPartitionsWorkersDiffers: Repartition only if the number of partitions and the number of workers (total number of cores) differs. Note however that even if the number of partitions is equal to the total number of cores, this does not guarantee that the correct number of DataSet objects is present in each partition: some partitions may be much larger or smaller than others.
* **repartitionStrategy**: Strategy by which repartitioning should be done
    * SparkDefault: This is the stardard repartitioning strategy used by Spark. Essentially, each object in the initial RDD is mapped to one of N RDDs independently at random. Consequently, the partitions may not be optimally balanced; this can be especially problematic with smaller RDDs, such as those used for preprocessed DataSet objects and frequent averaging periods (simply due to random sampling variation).
    * Balanced: This is a custom repartitioning strategy defined by DL4J. It attempts to ensure that each partition is more balanced (in terms of number of objects) compared to the SparkDefault option. However, in practice this requires an additional count operation to execute; in some cases (most notably in small networks, or those with a small amount of computation per minibatch), the benefit may not outweigh additional overhead of executing the better repartitioning.   
    



## Dependencies for Training on Spark

To use DL4J on Spark, you'll need to include the deeplearning4j-spark dependency:

```
        <dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>dl4j-spark_${scala.binary.version}</artifactId>
        <version>${dl4j.version}</version>
        </dependency>
```

Note that the ```_${scala.binary.version}``` should be ```_2.10``` or ```_2.11``` and should match the version of Spark you are using. 


## Spark Examples Repository

The [Deeplearning4j examples repo](https://github.com/deeplearning4j/dl4j-examples) ([old examples here](https://github.com/deeplearning4j/dl4j-spark-cdh5-examples)) contains a number of Spark examples.

## Using Intel MKL on Amazon Elastic MapReduce with Deeplearning4j

Releases of DL4J available on Maven Cental are distributed with OpenBLAS. Thus this section does not apply to users who are using using versions of Deeplearning4j on Maven Central.

If DL4J is built from source with Intel MKL as the BLAS library, some additional configuration is required to make this work on Amazon Elastic MapReduce.
When creating a cluster in EMR, to use Intel MKL it is necessary to provide some additional configuration.

Under the Create Cluster -> Advanced Options -> Edit Software Settings, add the following:

```
[
    {
        "Classification": "hadoop-env", 
        "Configurations": [
            {
                "Classification": "export", 
                "Configurations": [], 
                "Properties": {
                    "MKL_THREADING_LAYER": "GNU",
                    "LD_PRELOAD": "/usr/lib64/libgomp.so.1"
                }
            }
        ], 
        "Properties": {}
    }
]
```


## Resources

* [Deeplearning4j Examples Repo](https://github.com/deeplearning4j/dl4j-examples)
* ND4S: [N-Dimensional Arrays for Scala](https://github.com/deeplearning4j/nd4s)
* [ND4J, Scala & Scientific Computing](http://nd4j.org/scala.html)
* [Intro to Iterative Reduce](./iterativereduce)
