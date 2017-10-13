---
title: DeepLearning4j Introduction
layout: default
---

------
# Deep Learning, Defined

Deep learning encompasses both deep neural networks and deep reinforcement learning, which are subsets of machine learning, which itself is a subset of artifical intelligence. Broadly speaking, deep neural networks perform machine perception that extracts important features from raw data and makes some sort of prediction about each observation. Examples include identifying objects represented in images, mapping analog speech to written transcriptions, categorizing text by sentiment, and making forecasts about time series data.

Although neural networks were invented last century, only recently have they generated more excitement. Now that the computing ability to take advantage of the idea of neural networks exists, they have been used to set new, state of the art results in such fields as computer vision, natural language processing, and reinforcement learning. One well known accomplishment of deep learning was achieved by scientists at DeepMind who created a computer program called AlphaGo, which beat both a former world champion Go player and the current champion in 2016 and 2017, respectively. Many experts predicted this achievement would not come for another decade. 

There are many different kinds of neural networks, but the basic notion of how they work is simple. They are loosely based on the human brain and are comprised of one or more layers of "neurons", which are just mathematical operations that pass on a signal from the previous layer. At each layer, computations are applied on the input from neurons in the previous layer and the output is then relayed to the next layer. The output of the network's final layer will represent some prediction about the input data, depending on the task. The challenge in building a successful neural network is finding the right computations to apply at each layer. 

Neural networks can process high dimensional numerical and cateogorical data and perform tasks like regression, classification, clustering, and feature extraction. A neural network is created by first configuring its architecture based on the data and the task and then tuning its hyperparameters to optimize the performance of the neural network. Once the neural network has been trained and tuned sufficiently, it can be used to process new sets of data and return reasonably reliable predictions. 

To learn more about deep learning, check out these [resources](https://deeplearning4j.org/deeplearningforbeginners.html).

# Where DeepLearning4j Fits In

Eclipse Deeplearning4j (DL4J) is an open-source, JVM-based toolkit for building, training, and deploying neural networks. It was built to serve the Java and Scala communities and is user-friendly, stable, and well integrated with technologies such as Spark, CUDA, and cuDNN. Deeplearning4j also integrates with Python tools like Keras and TensorFlow to deploy their models to a production environment on the JVM.

Here are some reasons to use DeepLearning4j. 

- You are a data scientist in the field, or student with a Java project, and you need to integrate with a JVM stack (Hadoop, Spark, Kafka, ElasticSearch, Cassandra). You need to explore data, conduct and monitor experiments that apply various algorithms to the data and perform training on clusters to quickly obtain an accurate model for that data.  

- You are a data engineer or software developer in an enterprise environment who needs stable, reusable data pipelines and scalable and accurate predictions about the data. The use case here is to have data programmatically and automatically processed and analyzed to determine a designated result, using simple and understandable APIs.

# DeepLearning4j: System Requirements

- [**Java**](#java) 
- [**Apache Maven**](#maven) 
- [**Intellij IDEA**](#ide)  
- [**Git**](#git)
- [**CPU's/GPU's**](#gpu)

## <a name="java">Java</a>

The above tools should be installed for your machine before using DL4J. Java 1.7 or later (64 bit version) is required to run DL4J without any errors. In order to check which version of Java is installed on your computer, simply use the following command:

java -version

To install Java, download the [Java Development Kit here](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)

## <a name="maven">Maven</a>

Apache Maven, a dependency management and automated build tool for Java projects, is required as well. Maven will allow you to install project libraries easily and is well integrated with IDE's (Integrated Development Environment) such as IntelliJ. Dependencies can easily by edited within the project pom.xml file through Maven. Maven can be donwloaded [here](https://maven.apache.org/download.cgi) and follow their [instructions](https://maven.apache.org/install.html) depending on your machine. Alternatively, to install Maven on a Mac use the following command:

    brew install maven

Note that the previous command assumes [Homebrew](https://brew.sh/) is installed. For more information on Maven, please see our [Maven deep dive](#mavenoverview) below.

## <a name="ide">Intellij</a>

[Intellij](https://www.jetbrains.com/idea/download/) is a IDE that is recommended to use for DL4J although an IDE such as Eclipse may be used. Intellij will allow you to easily work with the DL4J API and is well integrated with Maven. Using Intellij will allow you to build neural networks in only a few steps. For more information on IntelliJ, please see our [IntelliJ deep dive](#intellij) below. 

## <a name="git">Git</a>

Lastly, you should have [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) installed on your machine. Git is a version control system that facilitates coordinating projects between multiple people. It can be used to save a project, keep track of changes, and facilitate distributed, non-linear workflows. You should use Git to build repositories to store your project code.

To update to the latest version of Git, use the following command: 

    git clone git://git.kernel.org/pub/scm/git/git.gi

## <a name="gpu">CPUs and GPUs</a>

DL4J can be used with both CPU's (central processing unit) and GPU's (graphics processing unit) which handle computer processing tasks. A GPU is composed of cores that can handle many threads simultaneously. In comparison to a CPU, GPU's are computationally more powerful and efficient and can accelerate applications of deep learning.

By default, CPU's are used with DL4J, but GPU's can be used with minor changes. To use GPU's for your application instead, the ND4J pom.xml file needs to be changed so that the fields of the ND4J dependency matches the following:

     groupId: org.nd4j
     artifactId: nd4j-cuda-7.5
     version: 0.9.1

Either Cuda 7.5 or 8.0 can be used to swap CPU's for GPU's. 

# <a name="component">DeepLearning4J Components</a>

Deeplearning4J is comprised of multiple components which all have different functionalities

The following are the main components of DL4J

- [**DataVec**](#datavec) performs data ingestion, normalization and transformation into feature vectors
- [**Deeplearning4j**](#dl4j) provides tools to configure neural networks 
- [**ND4J**](#nd4j) allows Java to access Native Libraries to quickly process Matrix Data on CPUs or GPUs.
- [**Arbiter**](#arbiter) helps search the hyperparameter space to find the best neural net configuration.
- [**RL4J**](#rl4j) implements Deep Q Learning, A3C and other reinforcement learning algorithms for the JVM.

## <a name="datavec">DataVec</a>

[DataVec](https://github.com/deeplearning4j/DataVec) is the main component used for preprocessing data to feed into neural networks. It is a library for machine learning ETL (extract, transform, and load) built by the Skymind team. The main purpose of DataVec is to convert raw data into data in a vector-like format suitable for training a neural network. DataVec currently supports formats such as CSV, raw text, images, and more. As a bonus, feature engineering, data cleaning, scaling, and normalization can be done using DataVec. 

## <a name="deeplearning4j">Deeplearning4J</a>

[Deeplearning4J](https://github.com/deeplearning4j/deeplearning4j) is the main component used to build neural networks. It provides the tools to configure, train, and evaluate neural networks. Examples of neural network code is contained in the DL4J-Examples repository of DL4J. This guide will also provide code examples in later chapters.  Deeplearning4J also contains a Keras Model Import subcomponent that assists with importing previously trained neural networks or model configurations from Keras into DL4J in MultiLayerNetwork and ComputationGraph format. Deeplearning4J is also broken up into other subcomponents that handle functionality for NLP, visualization, CUDA, and etc. 

## <a name="nd4j">ND4J</a>

[ND4J](https://github.com/deeplearning4j/nd4j) is the numerical processing library for Deeplearning4J. The main features of ND4J include verstaile n-dimensional array objects, multiplatform functionality including GPU's, and linear algebra / signal processing functions. This Java API is similar to that of Numpy and scikit-learn in Python. ND4J provides the functionality for the loss funcitons, optimization algorithms, and updaters for neural networks. For example,frequently used classes of ND4J include functions in GradientUpdater such as Gradient Descent, Adam, etch and BaseTransformOop such as sigmoid, relu, and tanh activation functions. 

## <a name="arbiter">Arbiter</a>

[Arbiter](https://github.com/deeplearning4j/Arbiter) is a library used for tuning hyperparameters of neural networks. This is often a computationally heavy challenge due to the large hyperparameter space. Currently, two hyperparameter optimization methods are supported in Arbiter: GridSearch and RandomSearch. These optimization methods search over different combinations of hyperparameters such as the learning rate and batch size in order to find the optimal combination. 

## <a name="rl4j">RL4J</a>

DL4J also provides support for doing reinforcement learning. These functionalities are contained in [RL4J](https://github.com/deeplearning4j/rl4j), a library and environment for Deep Q learning and other algorithms implemented in Java. RL4J is well integrated with DL4J and ND4J. 

# <a name="mavenoverview">Maven Overview</a>

Maven is a build automation and project management tool that provides a holistic build lifecycle framework for Java programmers. Users can manage builds, documentation, dependencies, releases and distribution using Maven. Maven contents are contained in the xml file and pom.xml, while its deliverable is a JAR (Java ARchive) file, which is a compressed package file that aggregates class files, metadata, and other resources of a project and helps Java runtimes deploy classes and resources.

## POM file

The POM file is a Project Object Model file, which resides in the project base directory. The pom.xml file contains information about the project such as the configuration details, goals, plugins, dependencies, project version, and build profiles. 

[Here](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/pom.xml) is an example of a pom.xml file. The pom.xml files all are required to have a project element and the groudID, artifactID, and version fields. These are located at the beginning of the example file. The groupID is  a unqiue ID of a project's group such as org.deeplearning4j; the artifactID is the ID or name of the project, such as deeplearning4j-examples-parent; finally the version is the version of the project used to distinguish separate versions of a project from one another. Each project requires at least one pom.xml file, which is usually at the root directory. If the main project is comprised of subprojects, the subprojects typically will each have a pom.xml file in their root directories. 

As mentioned earlier, Maven can be used to manage dependencies, which are external JAR files needed to run your project. This is especially useful if the project requires external API's or frameworks packaged in their own JAR files. If this is the case, it is important to keep up with correct versions of the external JAR files, which can become cumbersome to do. To manage dependencies in Maven, the pom.xml file is key. You can specify what external libraries and which versions the project depends on in the pom.xml file. Maven will then download these libraries and place them in your Maven local repository. Fortunatley, if the external libraries also require other libraries, Maven will download them into the local repository as well. In our [example](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/pom.xml), there are many dependencies the project depends on. For instance, ND4J is a library that is required. 

Each dependency requires its own groupID, artifactID and version fields. When Maven executes this pom.xml file, all the dependencies will be downloaded to your local Maven repository unless they are already present. You may be confused by the fact that the dependency in our example has a version specified by nd4j.version which is not defined by the project. This is because this file is a pom.xml file for a subproject, which inherits properties from the [base pom.xml file](https://github.com/deeplearning4j/dl4j-examples/blob/master/pom.xml) for the entire project. We can now see that nd4j.version is defined to be 0.9.1. 

## Maven Repositories

Maven repositories are essentially directories of packaged JAR files with metadata. There are three types of Maven respositories: local, central, and remote. If dependencies are specified, Maven looks for the libraries in the local, central, and remote repositories (if specified by the pom.xml file) in that order. The local repository is a directory on your computer, which contains downloaded dependencies. The central repository is located on the internet and is provided by the Maven community. Once the dependencies are found, they are downloaded into your local repository. Lastly, the remote repository is a repository on a web server or a local network. They can be used by an organization for hosting internal projects. If the dependencies are found in a remote repository, they are also downloaded into your local repository. You can specify a remote repository in the pom.xml file.

## Maven Build Cycles

Maven is also used to build a Java project. There are three types of build cycles (default, clean, and site), which are divided into build phases, which are further divided into build goals. Each build cycle is responsible for different aspects of building a Java project and are independently executed from one another. The default cycle handles compiling and packaging the project, the clean cycle removes temporary files, and the site cycle generates project documentation. 

A sequence of build phases comprise a build cycle. If a build phase is executed all previous build phases are executed by Maven as well. The default cycle is the most important since it builds the code. It has build phases validate (checks if all necessary information and dependencies are available), compile (compiles source code), test (runs tests against comiled source code), package (packages compiled code in a JAR), install (installs package into local repository), and deploy (copies package to remote repository). 

If a project requires specific functionality, Maven plug-ins can be created to add to the build process. This is done by creating a Java class that extends a Maven class. The plugin itself should reside in its own project. More information about plug-ins can be found [here](https://maven.apache.org/guides/mini/guide-configuring-plugins.html).

## Maven Troubleshooting

Older versions of Maven, such as 3.0.4, are likely to throw exceptions like a NoSuchMethodError. If this occures, simply upgrade to the latest version of Maven

If you receive a message such as ‘mvn is not recognised as an internal or external command, operable program or batch file,' you require Maven in your PATH variable.

If a Permgen error occurs during a DL4J build, you may need more heap space. You will need to alter your hidden .bash_profile file, which adds environmental variables to bash. Enter env in the command line to look at those variables. To add more heap space, enter this command in your console: echo “export MAVEN_OPTS=”-Xmx512m -XX:MaxPermSize=512m”” > ~/.bash_profile

For further reading, look at [Maven by Example](https://books.sonatype.com/mvnex-book/reference/public-book.html) and [Maven: The Complete Reference](https://books.sonatype.com/mvnref-book/reference/public-book.html).

# <a name="intellij">IntelliJ Overview and Example</a>

IntelliJ is our recommended IDE (integrated development environment) to use for DL4J. The free community version can be downloaded [here](https://www.jetbrains.com/idea/download/#section=mac). You can build projects within IntelliJ, which is well integrated with Maven. 

Here we will go over a quick example of using IntelliJ to run an example program from DL4J. This assumes IntelliJ and Git are already installed. First use the command line to enter the following:

    git clone https://github.com/deeplearning4j/dl4j-examples.git
    cd dl4j-examples/
    mvn clean install

Then open IntelliJ and import project and select "dl4j-examples." Next, choose "Import project from external model" and select Maven. The IntelliJ wizard will guide you through the various options. For the SDK select the jdk and choose finish. Once IntelliJ finishes downloading the required dependencies, you can run an example file on the left side.
