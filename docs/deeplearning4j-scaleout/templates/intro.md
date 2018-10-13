---
title: "Deeplearning4j on Spark: Introduction/Getting Started"
short_title: Introduction/Getting Started
description: "Deeplearning4j on Spark: Introduction"
category: Distributed Deep Learning
weight: 0
---

# Distributed Deep Learning with DL4J and Spark

Deeplearning4j supports neural network training on a cluster of CPU or GPU machines using Apache Spark along with support for distributed evaluation as well as distributed inference. 

### Multi-core vs multi-machine?

Users are encouraged to use DL4J’s Parallel-Wrapper implementation as shown in [this](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-cuda-specific-examples/src/main/java/org/deeplearning4j/examples/multigpu/MultiGpuLenetMnistExample.java) example before switching to training on multiple machines. Parallel-Wrappers allows for easy data parallel training of networks on a single machine with multiple cores. Distributed training on multiple machines incurs an overhead with necessary tasks like synchronization and network communication. For the benefits of parallelism to outweigh this overhead users should consider the ratio of network transfers to computation ensuring that the computation time is large enough to mask the additional overhead.


## DL4J’s Distributed Training Implementations

DL4J has two implementations of distributed training. 
  * Gradient sharing, available as of 1.0.0-beta: Based on [this](http://nikkostrom.com/publications/interspeech2015/strom_interspeech2015.pdf) paper by Nikko Strom, is an asynchronous SGD implementation with quantized and compressed updates implemented in Spark+Aeron
  * Parameter averaging: A synchronous SGD implementation with a single parameter server implemented entirely in Spark.


Users are directed towards the gradient sharing implementation which superseded the parameter averaging implementation. The gradient sharing implementation results in faster training times and is implemented to be scalable and fault-tolerant(as of 1.0.0-beta2). For the sake of completeness documentation will also cover the parameter averaging approach. The [technical reference section](deeplearning4j-scaleout-techinalref) covers details on the implementation.

In addition to distributed training DL4J also enables users to do distributed evaluation (including multiple evaluations simultaneously) and distributed inference. Refer to the [Deeplearning4j on Spark: How To Guides](deeplearning4j-scaleout-howto) for more details.

### Setup and Dependencies

To run training on GPUs make sure that you are specifying the correct backend in your pom file (nd4j-cuda-x.x for GPUs vs nd4j-native backend for CPUs) and have set up the machines with the appropriate CUDA libraries. Refer to the [Deeplearning4j on Spark: How To Guides](deeplearning4j-scaleout-howto) for more details.

To use the gradient sharing implementation include the following dependency:

```
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>dl4j-spark-parameterserver_${scala.binary.version}</artifactId>
    <version>${dl4j.spark.version}</version>
</dependency>
```

If using the parameter averaging implementation (**not** recommended) include:

```
<dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>dl4j-spark_${scala.binary.version}</artifactId>
        <version>${dl4j.spark.version}</version>
</dependency>
```
Note that ${scala.binary.version} is a Maven property with the value 2.10 or 2.11 and should match the version of Spark you are using.


## Key Concepts

The following are key classes the user should be familiar with to get started with distributed training with DL4J.

  * TrainingMaster: Specifies how distributed training will be conducted in practice i.e Parameter Averaging or Gradient Sharing.
  * SparkDl4jMultiLayer & SparkComputationGraph: These are wrappers around the MultiLayer and ComputationGraph classes in DL4J that enable the functionality related to distributed training. 
  * RDD<DataSet> & RDD<MultiDataSet>: An RDD with DL4J’s DataSet or MultiDataSet class. Note that the recommended best practice is to preprocess your data once, and save it to network storage such as HDFS. Refer to the [Deeplearning4j on Spark: How To Build Data Pipelines](deeplearning4j-scaleout-data-howto)section for more details.

## Minimal Examples
The following code snippets outlines the general setup required. The [API reference](deeplearning4j-scaleout-apiref) outlines detailed usage of the various classes. The user can submit a uber jar to Spark Submit for execution with the right options. See [Deeplearning4j on Spark: How To Guides](deeplearning4j-scaleout-howto) for further details.


### Gradient Sharing (Preferred Implementation)

```
JavaSparkContext sc = ...;
JavaRDD<DataSet> trainingData = ...;

//Model setup as on a single node. Either a MultiLayerConfiguration or a ComputationGraphConfiguration
MultiLayerConfiguration model = ...;

// Configure distributed training required for gradient sharing implementation
VoidConfiguration networkConf = VoidConfiguration.builder()
				.unicastPort(40123)
				.networkMask(“10.0.0.0/16”)
				.controllerAddress("10.0.2.4")
				.build()

//Create the TrainingMaster instance
TrainingMaster trainingMaster = new SharedTrainingMaster.Builder(networkConf)
				.batchSizePerWorker(batchSizePerWorker)
				.updatesThreshold(1e-3)
				.workersPerNode(numWorkersPerNode) // equal to GPUs or ~cores per node
                .meshBuildMode(MeshBuildMode.MESH)// or MeshBuildMode.PLAIN for < 32 nodes
				.build()

//Create the SparkDl4jMultiLayer instance
SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, model, trainingMaster)

//Execute training:
for (int i = 0; i < numEpochs; i++) {
    sparkNet.fit(trainingData);
    log.info("Completed Epoch {}", i);
}
```


### Parameter Averaging Implementation

```
JavaSparkContext sc = ...;
JavaRDD<DataSet> trainingData = ...;

//Model setup as on a single node. Either a MultiLayerConfiguration or a ComputationGraphConfiguration
MultiLayerConfiguration model = ...;

//Create the TrainingMaster instance
int examplesPerDataSetObject = 1;
TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
				.(other configuration options)
				.build();

//Create the SparkDl4jMultiLayer instance and fit the network using the training data:
SparkDl4jMultiLayer sparkNetwork = new SparkDl4jMultiLayer(sc, model, trainingMaster);

//Execute training:
for (int i = 0; i < numEpochs; i++) {
    sparkNet.fit(trainingData);
    log.info("Completed Epoch {}", i);
}
```

## Reference Examples
The [Deeplearning4j examples repo](https://github.com/deeplearning4j/dl4j-examples) contains a number of Spark examples that can be used by the user as reference.

## Further Reading

[Deeplearning4j on Spark: Technical Explanation](deeplearning4j-scaleout-techincalref)
[Deeplearning4j on Spark: How To Guides](deeplearning4j-scaleout-howto)
[Deeplearning4j on Spark: How To Build Data Pipelines](deeplearning4j-scaleout-data-howto)
[Deeplearning4j on Spark: API Reference](deeplearning4j-scaleout-apiref)
