<p align="center">
  <img src="https://www.zeljkoobrenovic.com/tools/tech/images/eclipse_deeplearning4j.png">
</p>

 [![Documentation](https://img.shields.io/badge/user-documentation-blue.svg)](https://deeplearning4j.konduit.ai/)
[![Get help at the community forum](https://img.shields.io/badge/Get%20Help-Community%20Forum-blue)](https://community.konduit.ai/)
[![javadoc](https://javadoc.io/badge2/org.deeplearning4j/deeplearning4j-nn/DL4J%20API%20Doc.svg)](https://javadoc.io/doc/org.deeplearning4j/deeplearning4j-nn)
[![javadoc](https://javadoc.io/badge2/org.nd4j/nd4j-api/ND4J%20API%20Doc.svg)](https://javadoc.io/doc/org.nd4j/nd4j-api)
[![License](https://img.shields.io/github/license/eclipse/deeplearning4j)](LICENSE)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/eclipse/deeplearning4j)


The **[Eclipse Deeplearning4J](https://deeplearning4j.konduit.ai/)** (DL4J) ecosystem is a set of projects intended to support all the needs of a JVM based deep learning application. This means starting with the raw data, loading and preprocessing it from wherever and whatever format it is in to building and tuning a wide variety of simple and complex deep learning networks. 

Because Deeplearning4J runs on the JVM you can use it with a wide variety of JVM based languages other than Java, like Scala, Kotlin, Clojure and many more. 

The DL4J stack comprises of:
- **DL4J**: High level API to build MultiLayerNetworks and ComputationGraphs with a variety of layers, including custom ones. Supports importing Keras models from h5, including tf.keras models (as of 1.0.0-beta7) and also supports distributed training on Apache Spark
- **ND4J**: General purpose linear algebra library with over 500 mathematical, linear algebra and deep learning operations. ND4J is based on the highly-optimized C++ codebase LibND4J that provides CPU (AVX2/512) and GPU (CUDA) support and acceleration by libraries such as OpenBLAS, OneDNN (MKL-DNN), cuDNN, cuBLAS, etc
- **SameDiff** : Part of the ND4J library, SameDiff is our automatic differentiation / deep learning framework. SameDiff uses a graph-based (define then run) approach, similar to TensorFlow graph mode. Eager graph (TensorFlow 2.x eager/PyTorch) graph execution is planned. SameDiff supports importing TensorFlow frozen model format .pb (protobuf) models. Import for ONNX, TensorFlow SavedModel and Keras models are planned. Deeplearning4j also has full SameDiff support for easily writing custom layers and loss functions.
- **DataVec**: ETL for machine learning data in a wide variety of formats and files (HDFS, Spark, Images, Video, Audio, CSV, Excel etc)
- **LibND4J** : C++ library that underpins everything. For more information on how the JVM acceses native arrays and operations refer to [JavaCPP](https://github.com/bytedeco/javacpp)
- **Python4J**: Bundled cpython execution for the JVM

All projects in the DL4J ecosystem support Windows, Linux and macOS. Hardware support includes CUDA GPUs (10.0, 10.1, 10.2 except OSX), x86 CPU (x86_64, avx2, avx512), ARM CPU (arm, arm64, armhf) and PowerPC (ppc64le).

## Community Support
For support for the project, please go over to https://community.konduit.ai/

## Using Eclipse Deeplearning4J in your project

Deeplearning4J has quite a few dependencies. For this reason we only support usage with a build tool.

```xml
<dependencies>
  <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-core</artifactId>
      <version>1.0.0-M2.1</version>
  </dependency>
  <dependency>
      <groupId>org.nd4j</groupId>
      <artifactId>nd4j-native-platform</artifactId>
      <version>1.0.0-M2.1</version>
  </dependency>
</dependencies>
```

Add these dependencies to your pom.xml file to use Deeplearning4J with the CPU backend. A full standalone project example is [available in the example repository](https://github.com/eclipse/deeplearning4j-examples), if you want to start a new Maven project from scratch.

## Code samples

Due to DL4J being a multi faceted project
with several modules in the mono repo, we recommend looking at the examples
for a taste of different usages of the different modules. Below
we'll link to examples for each module.

1. ND4J: https://github.com/deeplearning4j/deeplearning4j-examples/tree/master/nd4j-ndarray-examples
2. DL4J: https://github.com/deeplearning4j/deeplearning4j-examples/tree/master/dl4j-examples
3. Samediff: https://github.com/deeplearning4j/deeplearning4j-examples/tree/master/samediff-examples
4. Datavec: https://github.com/deeplearning4j/deeplearning4j-examples/tree/master/data-pipeline-examples
5. Python4j: https://deeplearning4j.konduit.ai/python4j/tutorials/quickstart


For users looking for being able to run models from other frameworks, see:
1. Onnx: https://github.com/deeplearning4j/deeplearning4j-examples/tree/master/onnx-import-examples
2. Tensorflow/Keras: https://github.com/deeplearning4j/deeplearning4j-examples/tree/master/tensorflow-keras-import-examples


## Documentation, Guides and Tutorials
You can find the official documentation for Deeplearning4J and the other libraries of its ecosystem at http://deeplearning4j.konduit.ai/.

## Want some examples?
We have separate repository with various examples available: https://github.com/eclipse/deeplearning4j-examples

## Building from source
It is preferred to use the official pre-compiled releases (see above). But if you want to build from source, first take a look at the prerequisites for building from source here: https://deeplearning4j.konduit.ai/multi-project/how-to-guides/build-from-source.

To build everything, we can use commands like
```
./change-cuda-versions.sh x.x
./change-scala-versions.sh 2.xx
./change-spark-versions.sh x
mvn clean install -Dmaven.test.skip -Dlibnd4j.cuda=x.x -Dlibnd4j.compute=xx
```
or
```
mvn -B -V -U clean install -pl  -Dlibnd4j.platform=linux-x86_64 -Dlibnd4j.chip=cuda -Dlibnd4j.cuda=11.0 -Dlibnd4j.compute=<your GPU CC> -Djavacpp.platform=linux-x86_64 -Dmaven.test.skip=true
```

An example of GPU "CC" or compute capability is 61 for Titan X Pascal.

## Running tests

In order to run tests, please see the platform-tests module.
This module only runs on jdk 11 (mostly due to spark and bugs with older scala versions + JDK 17)

platform-tests allows you to run dl4j for different backends.
There are a few properties you can specify on the command line:
1. backend.artifactId: this defaults to nd4j-native and will run tests on cpu,you can specify other backends like nd4j-cuda-11.6
2. dl4j.version: You can change the dl4j version that the tests run against. This defaults to 1.0.0-SNAPSHOT.

More parameters can be found here:
https://github.com/deeplearning4j/deeplearning4j/blob/c1bf8717e4839c8930e9c43183bf7b94d0cf84dc/platform-tests/pom.xml#L47





## Running project in Intellij IDEA:
1. Ensure you follow https://stackoverflow.com/questions/45370178/exporting-a-package-from-system-module-is-not-allowed-with-release on jdk 9 or later
2. Ignore all nd4j-shade submodules. Right click on each folder and click: Maven -> Ignore project


## License

[Apache License 2.0](LICENSE)


## Commercial Support
Deeplearning4J is actively developed by the team at [Konduit K.K.](https://konduit.ai). 

[If you need any commercial support feel free to reach out to us. at [support@konduit.ai](mailto:support@konduit.ai)  
