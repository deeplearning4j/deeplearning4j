---
title: "Deeplearning4j on Spark: How To Guides"
short_title: How To Guide
description: "Deeplearning4j on Spark: How To Guides"
category: Distributed Deep Learning
weight: 2
---

# Deeplearning4j on Spark: How To Guides

This page contains a number of how-to guides for common distributed training tasks.
Note that for guides on building data pipelines, see [here](deeplearning4j-scaleout-data-howto).

Before going through these guides, make sure you have read the introduction guide for deeplearning4j Spark training [here](deeplearning4j-scaleout-intro).

Before Training Guides
* [How to build an uber-JAR for training via Spark submit using Maven](#uberjar)
* [How to use GPUs for training on Spark](#gpus)
* [How to use CPUs on master, GPUs on the workers](#cpusgpus)
* [How to configure memory settings for Spark](#memory)
* [How to Configure Garbage Collection for Workers](#gc)
* [How to use Kryo Serialization with DL4J and ND4J](#kryo)
* [How to use YARN and GPUs](#yarngpus)
* [How to configure Spark Locality Configuration](#locality)

During and After Training Guides
* [How to configure encoding thresholds](#threshold)
* [How to perform distributed test set evaluation](#evaluation)
* [How to save (and load) neural networks trained on Spark](#saveload)
* [How to perform distributed inference](#inference)

Problems and Troubleshooting Guides
* [How to debug common Spark dependency problems (NoClassDefFoundExcption and similar)](#dependencyproblems)
* [How to fix "Error querying NTP server" errors](#ntperror)
* [How to Cache RDD[INDArray] and RDD[DataSet] Safely](#caching)
* [Fixing libgomp issues on Amazon Elastic MapReduce](#libgomp)
* [Failed training on Ubuntu 16.04 (Ubuntu bug that may affect DL4J Spark users)](#ubuntu16)

<br><br>

# Before Training - How-To Guides

## <a name="uberjar">How to build an uber-JAR for training via Spark submit using Maven</a>

When submitting a training job to a cluster, a typical workflow is to build an "uber-jar" that is submitted to Spark submit. An uber-jar is single JAR file containing all of the dependencies (libraries, class files, etc) required to run a job.
Note that Spark submit is a script that comes with a Spark distribution that users submit their job (in the form of a JAR file) to, in order to begin execution of their Spark job.

This guide assumes you already have code set up to train a network on Spark.

**Step 1: Decide on the required dependencies.**

There is a lot of overlap with single machine training with DL4J and ND4J. For example, for both single machine and Spark training you should include the standard set of deeplearning4j dependencies, such as:
* deeplearning4j-core
* deeplearning4j-spark
* nd4j-native-platform (for CPU-only training)

In addition, you will need to include the Deeplearning4j's Spark module, ```dl4j-spark_2.10``` or ```dl4j-spark_2.11```. This module is required for both development and execution of Deeplearning4j Spark jobs.
Be careful to use the spark version that matches your cluster - for both the Spark version (Spark 1 vs. Spark 2) and the Scala version (2.10 vs. 2.11). If these are mismatched, your job will likely fail at runtime.

Dependency example: Spark 2, Scala 2.11:
```
<dependency>
  <groupId>org.deeplearning4j</groupId>
  <artifactId>dl4j-spark_2.11</artifactId>
  <version>1.0.0-beta2_spark_2</version>
</dependency>
```

Depedency example, Spark 1, Scala 2.10:
```
<dependency>
  <groupId>org.deeplearning4j</groupId>
  <artifactId>dl4j-spark_2.10</artifactId>
  <version>1.0.0-beta2_spark_1</version>
</dependency>
```

Note that if you add a Spark dependency such as spark-core_2.11, this can be set to ```provided``` scope in your pom.xml (see [Maven docs](https://maven.apache.org/guides/introduction/introduction-to-dependency-mechanism.html#Dependency_Scope) for more details), as Spark submit will add Spark to the classpath. Adding this dependency is not required for execution on a cluster, but may be needed if you want to test or debug a Spark-based job on your local machine.


When training on CUDA GPUs, there are a couple of possible cases when adding CUDA dependencies:

**Case 1: Cluster nodes have CUDA toolkit installed on the master and worker nodes**

When the CUDA toolkit and CuDNN are available on the cluster nodes, we can use a smaller dependency:
* If the OS building the uber-jar is the same OS as the cluster: include nd4j-cuda-x.x
* If the OS building the uber-jar is different to the cluster OS (i.e., build on Windows, execute Spark on Linux cluster): include nd4j-cuda-x.x-platform
* In both cases, include 
where x.x is the CUDA version - for example, x.x=9.2 for CUDA 9.2.

**Case 2: Cluster nodes do NOT have the CUDA toolkit installed on the master and worker nodes**

When CUDA/CuDNN are NOT installed on the cluster nodes, we can do the following:
* First, include the dependencies as per 'Case 1' above
* Then include the "redist" javacpp-presets for the cluster operating system, as described here: [DL4J CuDNN Docs](./deeplearning4j-config-cudnn)


**Step 2: Configure your pom.xml file for building an uber-jar**

When using Spark submit, you will need an uber-jar to submit to start and run your job. After configuring the relevant dependencies in step 1, we need to configure the pom.xml file to properly build the uber-jar.

We recommend that you use the maven shade plugin for building an uber-jar. There are alternative tools/plugins for this purpose, but these do not always include all relevant files from the source jars, such as those required for Java's ServiceLoader mechanism to function correctly. (The ServiceLoader mechanism is used by ND4J and a lot of other software libraries).

A Maven shade configuration suitable for this purpose is provided in the example standalone sample project [pom.xml file](https://github.com/deeplearning4j/dl4j-examples/blob/master/standalone-sample-project/pom.xml):
```
<build>
    <plugins>
        <!-- Other plugins here if required -->

        <!-- Configure maven shade to produce an uber-jar when running "mvn package" -->
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-shade-plugin</artifactId>
            <version>${maven-shade-plugin.version}</version>
            <configuration>
                <shadedArtifactAttached>true</shadedArtifactAttached>
                <shadedClassifierName>bin</shadedClassifierName>
                <createDependencyReducedPom>true</createDependencyReducedPom>
                <filters>
                    <filter>
                        <artifact>*:*</artifact>
                        <excludes>
                            <exclude>org/datanucleus/**</exclude>
                            <exclude>META-INF/*.SF</exclude>
                            <exclude>META-INF/*.DSA</exclude>
                            <exclude>META-INF/*.RSA</exclude>
                        </excludes>
                    </filter>
                </filters>
            </configuration>

            <executions>
                <execution>
                    <phase>package</phase>
                    <goals>
                        <goal>shade</goal>
                    </goals>
                    <configuration>
                        <transformers>
                            <transformer implementation="org.apache.maven.plugins.shade.resource.AppendingTransformer">
                                <resource>reference.conf</resource>
                            </transformer>
                            <transformer implementation="org.apache.maven.plugins.shade.resource.ServicesResourceTransformer"/>
                            <transformer implementation="org.apache.maven.plugins.shade.resource.ManifestResourceTransformer">
                            </transformer>
                        </transformers>
                    </configuration>
                </execution>
            </executions>
        </plugin>
    </plugins>
</build>
```


**Step 3: Build the uber jar**

Finally, open up a command line window (bash on Linux, cmd on Windows, etc) simply run ```mvn package -DskipTests``` to build the uber-jar for your project.
Note that the uber-jar should be present under ```<project_root>/target/<project_name>-bin.jar```.
Be sure to use the large ```...-bin.jar``` file as this is the shaded jar with all of the dependencies.

That's is - you should now have an uber-jar that is suitable for submitting to spark-submit for training networks on Spark with CPUs or NVIDA (CUDA) GPUs.


<br><br>

## <a name="gpus">How to use GPUs for training on Spark</a>

Deeplearning4j and ND4J support GPU acceleration using NVIDA GPUs. DL4J Spark training can also be performed using GPUs.

DL4J and ND4J are designed in such a way that the code (neural network configuration, data pipeline code) is "backend independent". That is, you can write the code once, and execute it on either a CPU or GPU, simply by including the appropriate backend (nd4j-native backend for CPUs, or nd4j-cuda-x.x for GPUs). Executing on Spark is no different from executing on a single node in this respect: you need to simply include the appropriate ND4J backend, and make sure your machines (master/worker nodes in the case) are appropriately set with the CUDA libraries (see the [uber-jar guide](#uberjar) for running on CUDA without needing to install CUDA/cuDNN on each node).

When running on GPUs, there are a few components:
(a) The ND4J CUDA backend (nd4j-cuda-x.x dependency)
(b) The CUDA toolkit
(c) The Deeplearning4j CUDA dependency to gain cuDNN support (deeplearning4j-cuda-x.x)
(d) The cuDNN library files

Both (a) and (b) must be available for ND4J/DL4J to run using an available CUDA GPU run.
(c) and (d) are optional, though are recommended to get optimal performance - NVIDIA's cuDNN library is able to significantly speed up training for many layers, such as convolutional layers (ConvolutionLayer, SubsamplingLayer, BatchNormalization, etc) and LSTM RNN layers.

For configuring dependencies for Spark jobs, see the [uber-jar section](#uberjar) above.
For configuring cuDNN on a single node, see [Using Deeplearning4j with CuDNN](./deeplearning4j-config-cudnn)

<br><br>

## <a name="cpusgpus">How to use CPUs on master, GPUs on the workers</a>

In some cases, it may make sense to run the master using CPUs only, and the workers using GPUs.
If resources (i.e., the number of available GPU machines) are not constrained, it may simply be easier to have a homogeneous cluster: i.e., set up the cluster so that the master is using a GPU for execution also.

Assuming the master/driver is executing on a CPU machine, and the workers are executing on GPU machines, you can simply include both backends (i.e., both the ```nd4j-cuda-x.x``` and ```nd4j-native``` dependencies as described in the [uber-jar section](#uberjar)).

When multiple backends are present on the classpath, by default the CUDA backend will be tried first. If this cannot be loaded, the CPU (nd4j-native) backend will be loaded second. Thus, if the driver does not have a GPU, it should fall back to using a CPU. However, this default behaviour can be changed by setting the ```BACKEND_PRIORITY_CPU``` or ```BACKEND_PRIORITY_GPU``` environment variables on the master/driver, as described [here](https://github.com/deeplearning4j/deeplearning4j/blob/master/nd4j/nd4j-common/src/main/java/org/nd4j/config/ND4JEnvironmentVars.java).
The exact process for setting environment variables may depend on the cluster manager - Spark standalone vs. YARN vs. Mesos. Please consult the documentation for each on how to set the environment variables for Spark jobs for the driver/master.

<br><br>

## <a name="memory">How to configure memory settings for Spark</a>

For important background on how memory and memory configuration works for DL4J and ND4J, start by reading [Memory management for ND4J/DL4J](./deeplearning4j-config-memory).

The memory management on Spark is similar to memory management for single node training:
* On-heap memory is configured using the standard Java Xms and Xmx memory configuration settings
* Off-heap memory is configured using the javacpp system properties

However, memory configuration in the context of Spark adds some additional complications:
1. Often, memory configuration has to be done separately (sometimes using different mechanisms) for the driver/master vs. the workers
2. The approach for configuring memory can depend on the cluster resource manager - Spark standalone vs. YARN vs. Mesos, etc
3. Cluster resource manager default memory settings are often not appropriate for libraries (such as DL4J/ND4J) that rely heavily on off-heap memory

See the Spark documentation for your cluster manager:
* [YARN](https://spark.apache.org/docs/latest/running-on-yarn.html)
* [Mesos](https://spark.apache.org/docs/latest/running-on-mesos.html)
* [Spark Standalone](https://spark.apache.org/docs/latest/spark-standalone.html)

You should set 4 things:
1. The worker on-heap memory (Xmx) - usually set as an argument for Spark submit (for example, ```--executor-memory 4g``` for YARN)
2. The worker off-heap memory (javacpp system properties options)  (for example, ```--conf "spark.executor.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=8G"```)
3. The driver on-heap memory - usually set as an 
4. The driver off-heap memory


Some notes:
* On YARN, it is generally necessary to set the ```spark.driver.memoryOverhead``` and ```spark.executor.memoryOverhead``` properties. The default settings are much too small for DL4J training.
* On Spark standalone, you can also configure memory by modifying the ```conf/spark-env.sh``` file on each node, as described in the [Spark configuration docs](https://spark.apache.org/docs/latest/configuration.html#environment-variables). For example, you could add the following lines to set 8GB heap for the driver, 12 GB off-heap for the driver, 12GB heap for the workers, and 18GB off-heap for the workers:
    * ```SPARK_DRIVER_OPTS=-Dorg.bytedeco.javacpp.maxbytes=12G```
    * ```SPARK_DRIVER_MEMORY=8G```
    * ```SPARK_WORKER_OPTS=-Dorg.bytedeco.javacpp.maxbytes=18G```
    * ```SPARK_WORKER_MEMORY=12G```

All up, this might look like (for YARN, with 4GB on-heap, 5GB off-heap, 6GB YARN off-heap overhead):
```
--class my.class.name.here --num-executors 4 --executor-cores 8 --executor-memory 4G --driver-memory 4G --conf "spark.executor.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=5G" --conf "spark.driver.extraJavaOptions=-Dorg.bytedeco.javacpp.maxbytes=5G" --conf spark.yarn.executor.memoryOverhead=6144
```

<br><br>

## <a name="gc">How to Configure Garbage Collection for Workers</a>

One determinant of the performance of training is the frequency of garbage colection.
When using [Workspaces](https://deeplearning4j.org/docs/latest/deeplearning4j-config-memory) (see also [this](https://deeplearning4j.org/docs/latest/deeplearning4j-config-workspaces)), which are enabled by default, it can be helpful to reduce the frequency of garbage collection.
For simple machine training (and on the driver) this is easy:
```
// this will limit frequency of gc calls to 5000 milliseconds
Nd4j.getMemoryManager().setAutoGcWindow(5000)

// OR you could totally disable it
Nd4j.getMemoryManager().togglePeriodicGc(false);
```

However, setting this on the driver will not change the settings on the workers.
Instead, it can be set for the workers as follows:
```
new SharedTrainingMaster.Builder(voidConfiguration, minibatch)
    <other configuration>
    .workerTogglePeriodicGC(true)       //Periodic garbage collection is enabled...
    .workerPeriodicGCFrequency(5000)    //...and is configured to be performed every 5 seconds (every 5000ms)
    .build();
```


The default (as of 1.0.0-beta3) is to perform periodic garbage collection every 5 seconds on the workers.

<br><br>

## <a name="kryo">How to use Kryo Serialization with DL4J and ND4J</a>

Deeplearning4j and ND4J can utilize Kryo serialization, with appropriate configuration.
Note that due to the off-heap memory of INDArrays, Kryo will offer less of a performance benefit compared to using Kryo in other contexts.

To enable Kryo serialization, first add the [nd4j-kryo dependency](https://search.maven.org/search?q=nd4j-kryo):
```
<dependency>
  <groupId>org.nd4j</groupId>
  <artifactId>nd4j-kryo_2.11</artifactId>
  <version>${dl4j-version}</version>
</dependency>
```
where ```${dl4j-version}``` is the version used for DL4J and ND4J.

Then, at the start of your training job, add the following code:
```
    SparkConf conf = new SparkConf();
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
    conf.set("spark.kryo.registrator", "org.nd4j.Nd4jRegistrator");
```

Note that when using Deeplearning4j's SparkDl4jMultiLayer or SparkComputationGraph classes, a warning will be logged if the Kryo configuration is incorrect.

<br><br>

## <a name="yarngpus">How to use YARN and GPUs</a>

For DL4J, the only requirement for CUDA GPUs is to use the appropriate backend, with the appropriate NVIDIA libraries either installed on each node, or provided in the uber-JAR (see [Spark how-to guide](deeplearning4j-scaleout-howto) for more details).
For recent versions of YARN, some additional configuration may be required in some cases - see the [YARN GPU documentation](https://hadoop.apache.org/docs/r3.1.0/hadoop-yarn/hadoop-yarn-site/UsingGpus.html) for more details.

Earlier version of YARN (for example, 2.7.x and similar) did not support GPUs natively.
For these versions, it is possible to utilize node labels to ensure that jobs are scheduled onto GPU-only nodes. For more details, see the Hadoop Yarn [documentation](https://hadoop.apache.org/docs/r2.7.3/hadoop-yarn/hadoop-yarn-site/NodeLabel.html)

Note that YARN-specific memory configuration (see [memory how-to](deeplearning4j-scaleout-howto#memory)) is also required.

<br><br>

## <a name="locality">How to Configure Spark Locality Configuration</a>

Configuring Spark locality settings is an optional configuration option that can improve training performance.

The summary: adding ```--conf spark.locality.wait=0``` to your Spark submit configuration may marginally reduce training times, by scheduling the network fit operations to be started sooner.

For more details, see [link 1](https://spark.apache.org/docs/latest/tuning.html#data-locality) and [link 2](https://spark.apache.org/docs/latest/configuration.html#scheduling).

<br><br>

# During and After Training Guides

## <a name="threshold">How to Configure Encoding Thresholds</a>

Deeplearning4j's Spark implementation uses a threshold encoding scheme for sending parameter updates between nodes. This encoding scheme results in a small quantized message, which significantly reduces the network cost of communicating updates. See the [technical explanation page](./deeplearning4j-scaleout-technicalref) for more details on this encoding process.

This threshold encoding process introduces a "distributed training specific" hyperparameter - the encoding threshold.
Both too large thresholds and too small thresholds can result in sub-optimal performance:

* Large thresholds mean infrequent communication - too infrequent and convergence can suffer
* Small thresholds mean more frequent communication - but smaller changes are communicated at each step

The encoding threshold to be used is controlled by the [ThresholdAlgorithm](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/solvers/accumulation/encoding/ThresholdAlgorithm.java). The specific implementation of the ThresholdAlgorithm determines what threshold should be used.

The default behaviour for DL4J is to use [AdaptiveThresholdAlgorithm](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/solvers/accumulation/encoding/threshold/AdaptiveThresholdAlgorithm.java) which tries to keep the sparsity ratio in a certain range.
* The sparsity ratio is defined as numValues(encodedUpdate)/numParameters - 1.0 means fully dense (all values communicated), 0.0 means fully sparse (no values communicated)
* Larger thresholds mean more sparse values (less network communication), and a smaller threshold means less sparse values (more network communication)
* The AdaptiveThresholdAlgorithm tries to keep the sparsity ratio between 0.01 and 0.0001 by default. If the sparsity of the updates falls outside of this range, the threshold is either increased or decreased until it is within this range.
* An initial threshold value still needs to be set - we have found the

In practice, we have seen that this adaptive threshold process to work well.
The built-in implementations for threshold algorithms include:

* AdaptiveThresholdAlgorithm
* [FixedThresholdAlgorithm](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/solvers/accumulation/encoding/threshold/FixedThresholdAlgorithm.java): a fixed, non-adaptive threshold using the specified encoding threshold.
* [TargetSparsityThresholdAlgorithm](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/solvers/accumulation/encoding/threshold/TargetSparsityThresholdAlgorithm.java): an adaptive threshold algorithm that targets a specific sparsity, and increases or decreases the threshold to try to match the target.

In addition, DL4J has a [ResidualPostProcessor](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/solvers/accumulation/encoding/ResidualPostProcessor.java) interface, with the default implementation being [ResidualClippingPostProcessor](https://github.com/deeplearning4j/deeplearning4j/blob/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/optimize/solvers/accumulation/encoding/residual/ResidualClippingPostProcessor.java) which clips the residual vector to a maximum of 5x the current threshold, every 5 steps.
The motivation for this is that the "left over" parts of the updates (i.e., those parts not communicated) are store in the residual vector. If the updates are much larger than the threshold, we can have a phenomenon we have termed "residual explosion" - that is, the residual values can continue to grow to many times the threshold (hence would take many steps to communicate the gradient). The residual post processor is used to avoid this phenomenon.

The threshold algorithm (and initial threshold) and the residual post processor can be set as follows:
```
TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, minibatch)
    .thresholdAlgorithm(new AdaptiveThresholdAlgorithm(this.gradientThreshold))
    .residualPostProcessor(new ResidualClippingPostProcessor(5, 5))
    <other config>
    .build();
```

Finally, DL4J's SharedTrainingMaster also has an encoding debug mode, enabled by setting ```.encodingDebugMode(true)``` in the SharedTrainingmaster builder.
When this is enabled, each of the workers will log the current threshold, sparsity, and various other statistics about the encoding.
These statistics can be used to determine if the threshold is appropriately set: for example, many updates that are tens or hundreds of times the threshold may indicate the threshold is too low and should be increased; at the other end of the spectrum, very sparse updates (less than one in 10000 values being communicated) may indicate that the threshold should be decreased.

<br><br>

## <a name="evaluation">How to perform distributed test set evaluation</a>

Deeplearning4j supports most standard evaluation metrics for neural networks. For basic information on evaluation, see the [Deeplearning4j Evaluation Page](./deeplearning4j-nn-evaluation)

All of the [evaluation metrics](./deeplearning4j-nn-evaluation) that Deeplearning4j supports can be calculated in a distributed manner using Spark.

**Step 1: Prepare Your Data**

Evaluation data for Deeplearinng4j on Spark is very similar to training data. That is, you can use:
* ```RDD<DataSet>``` or ```JavaRDD<DataSet>``` for evaluating single input/output networks
* ```RDD<MultiDataSet>``` or ```JavaRDD<MultiDataSet>``` for evaluating multi input/output networks
* ```RDD<String>``` or ```JavaRDD<String>``` where each String is a path that points to serialized DataSet/MultiDataSet (or other minibatch file-based formats) on network storage such as HDFS.

See the data page (TODO: LINK) for details on how to prepare your data into one of these formats.

**Step 2: Prepare Your Network**

Creating your network is straightforward.
First, load your network (MultiLayerNetwork or ComputationGraph) into memory on the driver using the information from the following guide: [How to save (and load) neural networks trained on Spark](#saveload)

Then, simply create your network using:

```
JavaSparkContext sc = new JavaSparkContext();
MultiLayerNetwork net = <code to load network>
SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, cgForEval, null);
```

```
JavaSparkContext sc = new JavaSparkContext();
ComputationGraph net = <code to load network>
SparkComputationGraph sparkNet = new SparkComputationGraph(sc, net, null);
```

Note that you don't need to configure a TrainingMaster (i.e., the 3rd argument is null above), as evaluation does not use it.


**Step 3: Call the appropriate evaluation method**

For common cases, you can call one of the standard evalutation methods on SparkDl4jMultiLayer or SparkComputationGraph:
```
evaluate(RDD<DataSet>)                //Accuracy/F1 etc for classifiers
evaluate(JavaRDD<DataSet>)            //Accuracy/F1 etc for classifiers
evaluateROC(JavaRDD<DataSet>)         //ROC for single output binary classifiers
evaluateRegression(JavaRDD<DataSet>)  //For regression metrics
```

For performing multiple evaluations simultaneously (more efficient than performing them sequentially) you can use something like:
```
IEvaluation[] evaluations = new IEvaluation[]{new Evaluation(), new ROCMultiClass()};
JavaRDD<DataSet> data = ...;
sparkNet.doEvaluation(data, 32, evaluations);
```

Note that some of the evaluation methods have overloads with extra parameters, including:
* ```int evalNumWorkers``` - the number of evaluation workers - i.e., the number of copies of a network used for evaluation on each node (up to the maximum number of Spark threads per worker). For large networks (or limited cluster memory), you might want to reduce this to avoid running into memory problems.
* ```int evalBatchSize``` - the minibatch size to use when performing evaluation. This needs to be large enough to efficiently use the hardware resources, but small enough to not run out of memory. Values of 32-128 is unsually a good starting point; increase when more memory is available and for smaller networks; decrease if memory is a problem.
* ```DataSetLoader loader``` and ```MultiDataSetLoader loader``` - these are available when evaluating on a ```RDD<String>``` or ```JavaRDD<String>```. They are interfaces to load a path into a DataSet or MultiDataSet using a custom user-defined function. Most users will not need to use these, however the functionality is provided for greater flexibility. They would be used for example if the saved minibatch file format is not a DataSet/MultiDataSet but some other (possibly custom) format.


Finally, if you want to save the results of evaluation (of any type) you can save it to JSON format directly to remote storage such as HDFS as follows:
```
JavaSparkContext sc = new JavaSparkContext();
Evaluation eval = ...
String json = eval.toJson();
String writeTo = "hdfs:///output/directory/evaluation.json";
SparkUtils.writeStringToFile(writeTo, json, sc); //Also supports local file paths - file://
```
The import for ```SparkUtils``` is ```org.datavec.spark.transform.utils.SparkUtils```

The evaluation can be loaded using:
```
String json = SparkUtils.readStringFromFile(writeTo, sc);
Evaluation eval = Evaluation.fromJson(json);
```

<br><br>

## <a name="saveload">How to save (and load) neural networks trained on Spark</a>

Deeplearning4j's Spark functionality is built around the idea of wrapper classes - i.e., ```SparkDl4jMultiLayer``` and ```SparkComputationGraph``` internally use the standard ```MultiLayerNetwork``` and ```ComputationGraph``` classes.
You can access the internal MultiLayerNetwork/ComputationGraph classes using ```SparkDl4jMultiLayer.getNetwork()``` and ```SparkComputationGraph.getNetwork()``` respectively.

To save on the master/driver's local file system, get the network as described above and simply use the ```ModelSerializer``` class or ```MultiLayerNetwork.save(File)/.load(File)``` and ```ComputationGraph.save(File)/.load(File)``` methods.

To save to (or load from) a remote location or distributed file system such as HDFS, you can use input and output streams.

For example,
```
JavaSparkContext sc = new JavaSparkContext();
FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
String outputPath = "hdfs:///my/output/location/file.bin";
MultiLayerNetwork net = sparkNet.getNetwork();
try (BufferedOutputStream os = new BufferedOutputStream(fileSystem.create(new Path(outputPath)))) {
    ModelSerializer.writeModel(net, os, true);
}
```

Reading is a similar process:
```
JavaSparkContext sc = new JavaSparkContext();
FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration());
String outputPath = "hdfs:///my/output/location/file.bin";
MultiLayerNetwork net;
try(BufferedInputStream is = new BufferedInputStream(fileSystem.open(new Path(outputPath)))){
    net = ModelSerializer.restoreMultiLayerNetwork(is);
}
```

<br><br>


## <a name="inference">How to perform distributed inference</a>

Deeplearning4j's Spark implementation supports distributed inference. That is, we can easily generate predictions on an RDD of inputs using a cluster of machines.
This distributed inference can also be used for networks trained on a single machine and loaded for Spark (see the [saving/loading section](#saveload) for details on how to load a saved network for use with Spark).

Note: If you want to perform evaluation (i.e., calculate accuracy, F1, MSE, etc), refer to the [evaluation how-to](#evaluation) instead.

The method signatures for performing distributed inference are as follows:
```
SparkDl4jMultiLayer.feedForwardWithKey(JavaPairRDD<K, INDArray> featuresData, int batchSize) : JavaPairRDD<K, INDArray>
SparkComputationGraph.feedForwardWithKey(JavaPairRDD<K, INDArray[]> featuresData, int batchSize) : JavaPairRDD<K, INDArray[]>
```
There are also overloads that accept an input mask array, when required

Note the parameter ```K``` - this is a generic type to signify the unique 'key' used to identify each example. The key values are not used as part of the inference process. This key is required as Spark's RDDs are unordered - without this, we would have no way to know which element in the predictions RDD corresponds to which element in the input RDD.
The batch size parameter is used to specify the minibatch size when performing inference. It does not impact the values returned, but instead is used to balance memory use vs. computational efficiency: large batches might compute a little quicker overall, but require more memory. In many cases, a batch size of 64 is a good starting point to try if you are unsure of what to use.

<br><br><br>

# Problems and Troubleshooting Guides

## <a name="dependencyproblems">How to debug common Spark dependency problems (NoClassDefFoundExcption and similar)</a>

Unfortunately, dependency problems at runtime can occur on a cluster if your project is not configured correctly. These problems can occur with any Spark jobs, not just those using DL4J - and they may be caused by other dependencies or libraries on the classpath, not by Deeplearning4j dependencies.

When dependency problems occur, they typically produce exceptions like:
* NoSuchMethodException
* ClassNotFoundException
* AbstractMethodError

For example, mismatched Spark versions (trying to use Spark 1 on a Spark 2 cluster) can look like:
```
java.lang.AbstractMethodError: org.deeplearning4j.spark.api.worker.ExecuteWorkerPathMDSFlatMap.call(Ljava/lang/Object;)Ljava/util/Iterator;
```

Another class of errors is the ```UnsupportedClassVersionError``` for example ```java.lang.UnsupportedClassVersionError: XYZ : Unsupported major.minor version 52.0``` - this can result from trying to run (for example) Java 8 code on a cluster that is set up with only a Java 7 JRE/JDK.


How to debug dependency problems:

**Step 1: Collect Dependency Information**

The first step (when using Maven) is to produce a dependency tree that you can refer to.
Open a command line window (for example, bash on Linux, cmd on Windows), navigate to the root directory of your Maven project and run ```mvn dependency:tree```
This will give you a list of dependencies (direct and transient) that can be helpful to understand exactly what is on the classpath, and why.

Note also that ```mvn dependency:tree -Dverbose``` will provide extra information, and can be useful when debugging problems related to mismatched library versions.

**Step 2: Check your Spark Versions**

When running into dependency issues, check the following.

*First: check the Spark versions*
If your cluster is running Spark 2, you should be using a version of deeplearning4j-spark_2.10/2.11 (and DataVec) that ends with ```_spark_2```

Look through

If you find a problem, you should change your project dependencies as follows:
On a Spark 2 (Scala 2.11) cluster, use:
```
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>dl4j-spark_2.11</artifactId>
    <version>1.0.0-beta2_spark_2</version>
</dependency>
```
whereas on a Spark 1 (Scala 2.11) cluster, you should use:
```
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>dl4j-spark_2.11</artifactId>
    <version>1.0.0-beta2_spark_1</version>
</dependency>
```

**Step 3: Check the Scala Versions**

Apache Spark is distributed with versions that support both Scala 2.10 and Scala 2.11.

To avoid problems with Scala versions, you need to do two things:
(a) Ensure you don't have a mix of Scala 2.10 and Scala 2.11 (or 2.12) dependencies on your project classpath. Check your dependency tree for entries ending in ```_2.10``` or ```_2.11```: for example, ```org.apache.spark:spark-core_2.11:jar:1.6.3:compile``` is a Spark 1 (1.6.3) dependency using Scala 2.11
(b) Ensure that your project matches what the cluster is using. For example, if you cluster is running Spark 2 with Scala 2.11, all of your Scala dependencies should use 2.11 also. Note that Scala 2.11 is more common for Spark clusters.

If you find mismatched Scala versions, you will need to align them by changing the dependency versions in your pom.xml (or similar configuration file for other dependency management systems). Many libraries (including Spark and DL4J) release dependencies with both Scala 2.10 and 2.11 versions.

**Step 4: Check for Mismatched Library Versions**

A number of common utility libraries that are widely used across the Java ecosystem are not compatible across versions. For example, Spark might rely on library X version Y and will fail to run when library X version Z is on the classpath. Furthermore, many of these libraries are split into multiple modules (i.e., multiple separate modular dependencies) that won't work correctly when mixing different versions.

Some that can commonly cause problems include:
* Jackson
* Guava

DL4J and ND4J use versions of these libraries that should avoid dependency conflicts with Spark.
However, it is possible that other (3rd party libraries) can pull in versions of these dependencies.

Often, the exception will give a hint of where to look - i.e., the stack trace might include a specific class, which can be used to identify the problematic library.

**Step 5: Once Identified, Fix the Dependency Conflict**

To debug these sorts of problems, check the dependency tree (the output of ```mvn dependency:tree -Dverbose```) carefully. Where necessary, you can use [exclusions](https://maven.apache.org/guides/introduction/introduction-to-optional-and-excludes-dependencies.html) or add the problematic dependency as a direct dependency to force it's version in your probelm. To do this, you would add the dependency of the version you want directly to your project. Often, this is enough to solve the problem.

Keep in mind that when using Spark submit, Spark will add a copy of Spark and it's dependent libraries to the driver and worker classpaths.
This means that for dependencies that are added by Spark, you can't simply exclude them in your project - Spark submit will add them at runtime whether you exclude them or not in your project.

One additional setting that is worth knowing about is the (experimental) Spark configuration options, ```spark.driver.userClassPathFirst``` and ```spark.executor.userClassPathFirst``` (See the [Spark configuartion docs](https://spark.apache.org/docs/latest/configuration.html) for more details). In some cases, these options may be a fix for dependency issues.

<br><br>

## <a name="caching">How to Cache RDD[INDArray] and RDD[DataSet] Safely</a>

Spark has some issues regarding how it handles Java objects with large off-heap components, such as the DataSet and INDArray objects used in Deeplearning4j. This section explains the issues related to caching/persisting these objects.

The key points to know about are:

* MEMORY_ONLY and MEMORY_AND_DISK persistence can be problematic with off-heap memory, due to Spark not properly estimating the size of objects in the RDD. This can lead to out of (off-heap) memory issues.
* When persisting a ```RDD<DataSet>``` or ```RDD<INDArray>``` for re-use, use MEMORY_ONLY_SER or MEMORY_AND_DISK_SER

**Why MEMORY_ONLY_SER or MEMORY_AND_DISK_SER Are Recommended**

One of the way that Apache Spark improves performance is by allowing users to cache data in memory. This can be done using the ```RDD.cache()``` or ```RDD.persist(StorageLevel.MEMORY_ONLY())``` to store the contents in-memory, in deserialized (i.e., standard Java object) form.
The basic idea is simple: if you persist a RDD, you can re-use it from memory (or disk, depending on configuration) without having to recalculate it. However, large RDDs may not entirely fit into memory. In this case, some parts of the RDD have to be recomputed or loaded from disk, depending on the storage level used. Furthermore, to avoid using too much memory, Spark will drop parts (blocks) of an RDD when required.

The main storage levels available in Spark are listed below. For an explanation of  these, see the [Spark Programming Guide](https://spark.apache.org/docs/1.6.2/programming-guide.html#rdd-persistence).

* MEMORY_ONLY
* MEMORY_AND_DISK
* MEMORY_ONLY_SER
* MEMORY_AND_DISK_SER
* DISK_ONLY

The problem with Spark is how it handles memory. In particular, Spark will drop part of an RDD (a block) based on the estimated size of that block. The way Spark estimates the size of a block depends on the persistence level. For ```MEMORY_ONLY``` and ```MEMORY_AND_DISK``` persistence, this is done by walking the Java object graph - i.e., look at the fields in an object and recursively estimate the size of those objects. This process does not however take into account the off-heap memory used by Deeplearning4j or ND4J. For objects like DataSets and INDArrays (which are stored almost entirely off-heap), Spark significantly under-estimates the true size of the objects using this process. Furthermore, Spark considers only the amount of on-heap memory use when deciding whether to keep or drop blocks. Because DataSet and INDArray objects have a very small on-heap size, Spark will keep too many of them around with ```MEMORY_ONLY``` and ```MEMORY_AND_DISK``` persistence, resulting in off-heap memory being exhausted, causing out of memory issues.

However, for ```MEMORY_ONLY_SER``` and ```MEMORY_AND_DISK_SER``` Spark stores blocks in *serialized* form, on the Java heap. The size of objects stored in serialized form can be estimated accurately by Spark (there is no off-heap memory component for the serialized objects) and consequently Spark will drop blocks when required - avoiding any out of memory issues.

<br><br>

## <a name="ntperror">How to fix "Error querying NTP server" errors</a>

DL4J's parameter averaging implementation has the option to collect training stats, by using ```SparkDl4jMultiLayer.setCollectTrainingStats(true)```.
When this is enabled, internet access is required to connect to the NTP (network time protocal) server.

It is possible to get errors like ```NTPTimeSource: Error querying NTP server, attempt 1 of 10```. Sometimes these failures are transient (later retries will work) and can be ignored. However, if the Spark cluster is configured such that one or more of the workers cannot access the internet (or specifically, the NTP server), all retries can fail.

Two solutions are available:

1. Don't use ```sparkNet.setCollectTrainingStats(true)``` - this functionality is optional (not required for training), and is disabled by default
2. Set the system to use the local machine clock instead of the NTP server, as the time source (note however that the timeline information may be very inaccurate as a result)
To use the system clock time source, add the following to Spark submit:
```
--conf spark.driver.extraJavaOptions=-Dorg.deeplearning4j.spark.time.TimeSource=org.deeplearning4j.spark.time.SystemClockTimeSource
--conf spark.executor.extraJavaOptions=-Dorg.deeplearning4j.spark.time.TimeSource=org.deeplearning4j.spark.time.SystemClockTimeSource
```

<br><br>

## <a name="ubuntu16">Failed training on Ubuntu 16.04 (Ubuntu bug that may affect DL4J users)</a>

When running a Spark on YARN cluster on Ubuntu 16.04 machines, chances are that after finishing a job, all processes owned by the user running Hadoop/YARN are killed. This is related to a bug in Ubuntu, which is documented at https://bugs.launchpad.net/ubuntu/+source/procps/+bug/1610499. There's also a Stackoverflow discussion about it at http://stackoverflow.com/questions/38419078/logouts-while-running-hadoop-under-ubuntu-16-04.

Some workarounds are suggested. 

**Option 1**

Add
```
[login]
KillUserProcesses=no
```
to /etc/systemd/logind.conf, and reboot.

**Option 2**

Copy the /bin/kill binary from Ubuntu 14.04 and use that one instead. 

**Option 3**

Downgrade to Ubuntu 14.04 

**Option 4**

## <a href="caching">How to Cache RDD[INDArray] and RDD[DataSet] Safely</a>

Spark has some issues regarding how it handles Java objects with large off-heap components, such as the DataSet and INDArray objects used in Deeplearning4j. This section explains the issues related to caching/persisting these objects.

The key points to know about are:

* MEMORY_ONLY and MEMORY_AND_DISK persistence can be problematic with off-heap memory, due to Spark not properly estimating the size of objects in the RDD. This can lead to out of (off-heap) memory issues.
* When persisting a ```RDD<DataSet>``` or ```RDD<INDArray>``` for re-use, use MEMORY_ONLY_SER or MEMORY_AND_DISK_SER

**Why MEMORY_ONLY_SER or MEMORY_AND_DISK_SER Are Recommended**

One of the way that Apache Spark improves performance is by allowing users to cache data in memory. This can be done using the ```RDD.cache()``` or ```RDD.persist(StorageLevel.MEMORY_ONLY())``` to store the contents in-memory, in deserialized (i.e., standard Java object) form.
The basic idea is simple: if you persist a RDD, you can re-use it from memory (or disk, depending on configuration) without having to recalculate it. However, large RDDs may not entirely fit into memory. In this case, some parts of the RDD have to be recomputed or loaded from disk, depending on the storage level used. Furthermore, to avoid using too much memory, Spark will drop parts (blocks) of an RDD when required.

The main storage levels available in Spark are listed below. For an explanation of  these, see the [Spark Programming Guide](https://spark.apache.org/docs/1.6.2/programming-guide.html#rdd-persistence).

* MEMORY_ONLY
* MEMORY_AND_DISK
* MEMORY_ONLY_SER
* MEMORY_AND_DISK_SER
* DISK_ONLY

The problem with Spark is how it handles memory. In particular, Spark will drop part of an RDD (a block) based on the estimated size of that block. The way Spark estimates the size of a block depends on the persistence level. For ```MEMORY_ONLY``` and ```MEMORY_AND_DISK``` persistence, this is done by walking the Java object graph - i.e., look at the fields in an object and recursively estimate the size of those objects. This process does not however take into account the off-heap memory used by Deeplearning4j or ND4J. For objects like DataSets and INDArrays (which are stored almost entirely off-heap), Spark significantly under-estimates the true size of the objects using this process. Furthermore, Spark considers only the amount of on-heap memory use when deciding whether to keep or drop blocks. Because DataSet and INDArray objects have a very small on-heap size, Spark will keep too many of them around with ```MEMORY_ONLY``` and ```MEMORY_AND_DISK``` persistence, resulting in off-heap memory being exhausted, causing out of memory issues.

However, for ```MEMORY_ONLY_SER``` and ```MEMORY_AND_DISK_SER``` Spark stores blocks in *serialized* form, on the Java heap. The size of objects stored in serialized form can be estimated accurately by Spark (there is no off-heap memory component for the serialized objects) and consequently Spark will drop blocks when required - avoiding any out of memory issues.

## <a href="ntperror">How to fix "Error querying NTP server" errors</a>

DL4J's parameter averaging implementation has the option to collect training stats, by using ```SparkDl4jMultiLayer.setCollectTrainingStats(true)```.
When this is enabled, internet access is required to connect to the NTP (network time protocal) server.

It is possible to get errors like ```NTPTimeSource: Error querying NTP server, attempt 1 of 10```. Sometimes these failures are transient (later retries will work) and can be ignored. However, if the Spark cluster is configured such that one or more of the workers cannot access the internet (or specifically, the NTP server), all retries can fail.

Two solutions are available:

1. Don't use ```sparkNet.setCollectTrainingStats(true)``` - this functionality is optional (not required for training), and is disabled by default
2. Set the system to use the local machine clock instead of the NTP server, as the time source (note however that the timeline information may be very inaccurate as a result)
To use the system clock time source, add the following to Spark submit:
```
--conf spark.driver.extraJavaOptions=-Dorg.deeplearning4j.spark.time.TimeSource=org.deeplearning4j.spark.time.SystemClockTimeSource
--conf spark.executor.extraJavaOptions=-Dorg.deeplearning4j.spark.time.TimeSource=org.deeplearning4j.spark.time.SystemClockTimeSource
```

## <a href="ubuntu16">Failed training on Ubuntu 16.04 (Ubuntu bug that may affect DL4J users)</a>

When running a Spark on YARN cluster on Ubuntu 16.04 machines, chances are that after finishing a job, all processes owned by the user running Hadoop/YARN are killed. This is related to a bug in Ubuntu, which is documented at https://bugs.launchpad.net/ubuntu/+source/procps/+bug/1610499. There's also a Stackoverflow discussion about it at http://stackoverflow.com/questions/38419078/logouts-while-running-hadoop-under-ubuntu-16-04.

Some workarounds are suggested. 

**Option 1**

Add
```
[login]
KillUserProcesses=no
```
to /etc/systemd/logind.conf, and reboot.

**Option 2**

Copy the /bin/kill binary from Ubuntu 14.04 and use that one instead. 

**Option 3**

Downgrade to Ubuntu 14.04 

**Option 4**

run ```sudo loginctl enable-linger hadoop_user_name``` on cluster nodes