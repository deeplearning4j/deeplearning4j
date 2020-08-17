<p align="center">
  <img src="https://www.zeljkoobrenovic.com/tools/tech/images/eclipse_deeplearning4j.png">
</p>

 [![Documentation](https://img.shields.io/badge/user-documentation-blue.svg)](https://deeplearning4j.konduit.ai/)
[![Get help at the community forum](https://img.shields.io/badge/Get%20Help-Community%20Forum-blue)](https://community.konduit.ai/)
[![javadoc](https://javadoc.io/badge2/org.deeplearning4j/deeplearning4j-nn/DL4J%20API%20Doc.svg)](https://javadoc.io/doc/org.deeplearning4j/deeplearning4j-nn)
[![javadoc](https://javadoc.io/badge2/org.nd4j/nd4j-api/ND4J%20API%20Doc.svg)](https://javadoc.io/doc/org.nd4j/nd4j-api)
[![License](https://img.shields.io/github/license/eclipse/deeplearning4j)](LICENSE)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/konduitai/deeplearning4j)


The **[Eclipse Deeplearning4J](https://deeplearning4j.konduit.ai/)** (DL4J) ecosystem is a set of projects intended to support all the needs of a JVM based deep learning application. This means starting with the raw data, loading and preprocessing it from wherever and whatever format it is in to building and tuning a wide variety of simple and complex deep learning networks. 

Because Deeplearning4J runs on the JVM you can use it with a wide variety of JVM based languages other than Java, like Scala, Kotlin, Clojure and many more. 

The DL4J stack comprises of:
- **DL4J**: High level API to build MultiLayerNetworks and ComputationGraphs with a variety of layers, including custom ones. Supports importing Keras models from h5, including tf.keras models (as of 1.0.0-beta7) and also supports distributed training on Apache Spark
- **ND4J**: General purpose linear algebra library with over 500 mathematical, linear algebra and deep learning operations. ND4J is based on the highly-optimized C++ codebase LibND4J that provides CPU (AVX2/512) and GPU (CUDA) support and acceleration by libraries such as OpenBLAS, OneDNN (MKL-DNN), cuDNN, cuBLAS, etc
- **SameDiff** : Part of the ND4J library, SameDiff is our automatic differentiation / deep learning framework. SameDiff uses a graph-based (define then run) approach, similar to TensorFlow graph mode. Eager graph (TensorFlow 2.x eager/PyTorch) graph execution is planned. SameDiff supports importing TensorFlow frozen model format .pb (protobuf) models. Import for ONNX, TensorFlow SavedModel and Keras models are planned. Deeplearning4j also has full SameDiff support for easily writing custom layers and loss functions.
- **DataVec**: ETL for machine learning data in a wide variety of formats and files (HDFS, Spark, Images, Video, Audio, CSV, Excel etc)
- **Arbiter**: Library for hyperparameter search
- **LibND4J** : C++ library that underpins everything. For more information on how the JVM acceses native arrays and operations refer to [JavaCPP](https://github.com/bytedeco/javacpp)

All projects in the DL4J ecosystem support Windows, Linux and macOS. Hardware support includes CUDA GPUs (10.0, 10.1, 10.2 except OSX), x86 CPU (x86_64, avx2, avx512), ARM CPU (arm, arm64, armhf) and PowerPC (ppc64le).

## Using Eclipse Deeplearning4J in your project

Deeplearning4J has quite a few dependencies. For this reason we only support usage with a build tool.

```xml
<dependencies>
  <dependency>
      <groupId>org.deeplearning4j</groupId>
      <artifactId>deeplearning4j-core</artifactId>
      <version>1.0.0-beta7</version>
  </dependency>
  <dependency>
      <groupId>org.nd4j</groupId>
      <artifactId>nd4j-native-platform</artifactId>
      <version>1.0.0-beta7</version>
  </dependency>
</dependencies>
```

Add these dependencies to your pom.xml file to use Deeplearning4J with the CPU backend. A full standalone project example is [available in the example repository](https://github.com/eclipse/deeplearning4j-examples), if you want to start a new Maven project from scratch.

## A taste of code
Deeplearning4J offers a very high level API for defining even complex neural networks. The following example code shows
you how LeNet, a convolutional neural network, is defined in DL4J.

```java
MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1,1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(28,28,1))
                .build();

```

## Documentation, Guides and Tutorials
You can find the official documentation for Deeplearning4J and the other libraries of its ecosystem at http://deeplearning4j.konduit.ai/.

## Want some examples?
We have separate repository with various examples available: https://github.com/eclipse/deeplearning4j-examples

## Building from source
It is preferred to use the official pre-compiled releases (see above). But if you want to build from source, first take a look at the prerequisites for building from source here: https://deeplearning4j.konduit.ai/getting-started/build-from-source.

To build everything, we can use commands like
```
./change-cuda-versions.sh x.x
./change-scala-versions.sh 2.xx
./change-spark-versions.sh x
mvn clean install -Dmaven.test.skip -Dlibnd4j.cuda=x.x -Dlibnd4j.compute=xx
```
or
```
mvn -B -V -U clean install -pl '!jumpy,!pydatavec,!pydl4j' -Dlibnd4j.platform=linux-x86_64 -Dlibnd4j.chip=cuda -Dlibnd4j.cuda=9.2 -Dlibnd4j.compute=<your GPU CC> -Djavacpp.platform=linux-x86_64 -Dmaven.test.skip=true
```

An example of GPU "CC" or compute capability is 61 for Titan X Pascal.


## License

[Apache License 2.0](LICENSE)


## Commercial Support
Deeplearning4J is actively developed by the team at [Konduit K.K.](http://www.konduit.ai). 

[If you need any commercial support feel free to reach out to us.](https://konduit.ai/konduit-open-source-support/)  
