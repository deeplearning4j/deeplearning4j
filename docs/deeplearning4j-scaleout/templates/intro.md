---
title: "Deeplearning4j on Spark: Introduction/Getting Started"
short_title: Introduction/Getting Started
description: "Deeplearning4j on Spark: Introduction"
category: Distributed Deep Learning
weight: 0
---

# Distributed Deep Learning with DL4J and Spark

Deeplearning4j supports neural network training on a cluster of CPU or GPU machines using Apache Spark. Deeplearning4j also supports distributed evaluation as well as distributed inference using Spark.

## DL4J’s Distributed Training Implementations

DL4J has two implementations of distributed training. 
  * Gradient sharing, available as of 1.0.0-beta: Based on [this](http://nikkostrom.com/publications/interspeech2015/strom_interspeech2015.pdf) paper by Nikko Strom, is an asynchronous SGD implementation with quantized and compressed updates implemented in Spark+Aeron
  * Parameter averaging: A synchronous SGD implementation with a single parameter server implemented entirely in Spark.


Users are directed towards the gradient sharing implementation which superseded the parameter averaging implementation. The gradient sharing implementation results in faster training times and is implemented to be scalable and fault-tolerant (as of 1.0.0-beta3). For the sake of completeness, this page will also cover the parameter averaging approach. The [technical reference section](deeplearning4j-scaleout-technicalref) covers details on the implementation.

In addition to distributed training DL4J also enables users to do distributed evaluation (including multiple evaluations simultaneously) and distributed inference. Refer to the [Deeplearning4j on Spark: How To Guides](deeplearning4j-scaleout-howto) for more details.

### When to use Spark for Training Neural Networks

Spark is not always the most appropriate tool for training neural networks.

You should use Spark when:
1. You have a cluster of machines for training (not just a single machine - this includes multi-GPU machines)
2. You need more than single machine to train the network
3. Your network is large to justify a distributed implementation

For a single machine with multiple GPUs or multiple physical processors, users should consider using DL4J's Parallel-Wrapper implementation as shown in [this example](https://github.com/eclipse/deeplearning4j-examples/blob/master/dl4j-cuda-specific-examples/src/main/java/org/deeplearning4j/examples/multigpu/MultiGpuLenetMnistExample.java). ParallelWrapper allows for easy data parallel training of networks on a single machine with multiple cores. Spark has higher overheads compared to ParallelWrapper for single machine training.

Similarly, if you don't need Spark (smaller networks and/or datasets) - it is recommended to use single machine training, which is usually simpler to set up.

For a network to be large enough: here's a rough guide. If the network takes 100ms or longer to perform one iteration (100ms per fit operation on each minibatch), distributed training should work well with good scalability. At 10ms per iteration, we might expect sub-linear scaling of performance vs. number of nodes. At around 1ms or below per iteration, the communication overhead may be too much: training on a cluster may be no faster (or perhaps even slower) than on a single machine.
For the benefits of parallelism to outweigh the communication overhead, users should consider the ratio of network transfer time to computation time and ensure that the computation time is large enough to mask the additional overhead of distributed training.

### Setup and Dependencies

To run training on GPUs make sure that you are specifying the correct backend in your pom file (nd4j-cuda-x.x for GPUs vs nd4j-native backend for CPUs) and have set up the machines with the appropriate CUDA libraries. Refer to the [Deeplearning4j on Spark: How To Guides](deeplearning4j-scaleout-howto) for more details.

To use the gradient sharing implementation include the following dependency:

```
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>dl4j-spark-parameterserver_${scala.binary.version}</artifactId>
    <version>${dl4j.version}</version>
</dependency>
```

If using the parameter averaging implementation (again, the gradient sharing implemention should be preferred) include:

```
<dependency>
        <groupId>org.deeplearning4j</groupId>
        <artifactId>dl4j-spark_${scala.binary.version}</artifactId>
        <version>${dl4j.version}</version>
</dependency>
```
Note that ${scala.binary.version} is a Maven property with the value 2.10 or 2.11 and should match the version of Spark you are using.

## Key Concepts

The following are key classes the user should be familiar with to get started with distributed training with DL4J.

  * **TrainingMaster**: Specifies how distributed training will be conducted in practice. Implementations include Gradient Sharing (SharedTrainingMaster) or Parameter Averaging (ParameterAveragingTrainingMaster)
  * **SparkDl4jMultiLayer and SparkComputationGraph**: These are wrappers around the MultiLayerNetwork and ComputationGraph classes in DL4J that enable the functionality related to distributed training. For training, they are configured with a TrainingMaster.
  * **```RDD<DataSet>``` and ```RDD<MultiDataSet>```**: A Spark RDD with DL4J's DataSet or MultiDataSet classes define the source of the training data (or evaluation data). Note that the recommended best practice is to preprocess your data once, and save it to network storage such as HDFS. Refer to the [Deeplearning4j on Spark: How To Build Data Pipelines](deeplearning4j-scaleout-data-howto) section for more details.


The training workflow usually proceeds as follows:
1. Prepare training code with a few components:
    a. Neural network configuration
    b. Data pipeline
    c. SparkDl4jMultiLayer/SparkComputationGraph plus Trainingmaster
2. Create uber-JAR file (see [Spark how-to guide](deeplearning4j-scaleout-howto) for details)
3. Determine the arguments (memory, number of nodes, etc) for Spark submit
4. Submit the uber-JAR to Spark submit with the required arguments


## Minimal Examples
The following code snippets outlines the general setup required. The [API reference](deeplearning4j-scaleout-apiref) outlines detailed usage of the various classes. The user can submit a uber jar to Spark Submit for execution with the right options. See [Deeplearning4j on Spark: How To Guides](deeplearning4j-scaleout-howto) for further details.


### Gradient Sharing (Preferred Implementation)

```
JavaSparkContext sc = ...;
JavaRDD<DataSet> trainingData = ...;

//Model setup as on a single node. Either a MultiLayerConfiguration or a ComputationGraphConfiguration
MultiLayerConfiguration model = ...;

// Configure distributed training required for gradient sharing implementation
VoidConfiguration conf = VoidConfiguration.builder()
				.unicastPort(40123)             //Port that workers will use to communicate. Use any free port
				.networkMask(“10.0.0.0/16”)     //Network mask for communication. Examples 10.0.0.0/24, or 192.168.0.0/16 etc
				.controllerAddress("10.0.2.4")  //IP of the master/driver
				.build();

//Create the TrainingMaster instance
TrainingMaster trainingMaster = new SharedTrainingMaster.Builder(conf)
				.batchSizePerWorker(batchSizePerWorker) //Batch size for training
				.updatesThreshold(1e-3)                 //Update threshold for quantization/compression. See technical explanation page
				.workersPerNode(numWorkersPerNode)      // equal to number of GPUs. For CPUs: use 1; use > 1 for large core count CPUs
                .meshBuildMode(MeshBuildMode.MESH)      // or MeshBuildMode.PLAIN for < 32 nodes
				.build();

//Create the SparkDl4jMultiLayer instance
SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, model, trainingMaster);

//Execute training:
for (int i = 0; i < numEpochs; i++) {
    sparkNet.fit(trainingData);
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
}
```

## Further Reading

* [Deeplearning4j on Spark: Technical Explanation](deeplearning4j-scaleout-technicalref)
* [Deeplearning4j on Spark: How To Guides](deeplearning4j-scaleout-howto)
* [Deeplearning4j on Spark: How To Build Data Pipelines](deeplearning4j-scaleout-data-howto)
* [Deeplearning4j on Spark: API Reference](deeplearning4j-scaleout-apiref)
* The [Deeplearning4j examples repo](https://github.com/eclipse/deeplearning4j-examples) contains a number of Spark examples that can be used by the user as reference.
