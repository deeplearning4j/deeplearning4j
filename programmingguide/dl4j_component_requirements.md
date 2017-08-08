title: DeepLearning4j: Component Requirements
layout: default

------

# DeepLearning4j: Components and Requirements

The DeepLearning4j suggested options.

## Required Tools

(https://deeplearning4j.org/quickstart#prerequisites)

- **Java** — Used to enhance security, durability. It is backward compatible. Errors are caught at compile and not at runtime. 
- **Apache Maven** — Used for dependency management. It is an automated build tool for Java projects. Edit the required depencdies in the pom.xml file through Maven.
- **Intellij IDEA** — Used as the Itegrated Development Environment (IDE) for creating the neural network models in Java. 
- **Git** — Used as the repository and as a community communicaton channel.

## CPU and GPU

(http://nd4j.org/gpu_native_backends.html )

What a brief discussion/one-two sentences on advantages, differences GPU over CPU. 

## Java vs Python

Describe how Java differs from Python for implementing deep learning. Particularly the aspect of nesting and applying dependencies through the pom.xml file.  

## DeepLearning4J Components

(https://deeplearning4j.org/overview#deeplearning4j-components)

Any requirements in addition to the tools listed above to run these?

- **DataVec** performs data ingestion, normalization and transformation into feature vectors
- **DeepLearning4j** provides tools to configure neural networks and build computation graphs
- **DL4J-Examples** contains working examples for classification and clustering of images, time series and text.
- **Keras Model Import** helps import trained models from Python and Keras to DeepLearning4J and Java.
- **ND4J** allows Java to access Native Libraries to quickly process Matrix Data on CPUs or GPUs.
- **ScalNet** is a Scala wrapper for Deeplearning4j inspired by Keras. Runs on multi-GPUs with Spark.
- **RL4J** implements Deep Q Learning, A3C and other reinforcement learning algorithms for the JVM.
- **Arbiter** helps search the hyperparameter space to find the best neural net configuration.



 

