Deeplearning4j Tutorials
========================

Welcome to the [Deeplearning4j](https://deeplearning4j.org/) tutorial series in Zeppelin. This README will help you get started with using Zeppelin notebooks and loading the required dependencies.

## Prerequisites

While Deeplearning4j is written in Java, the Java Virtual Machine (JVM) lets you import and share code in other JVM languages. These tutorials are written in Scala, the de facto standard for data science in the Java environment. There's nothing stopping you from using any other interpreter such as Java, Kotlin, or Clojure.

If you're coming from non-JVM languages like Python or R, you may want to read about how the JVM works before using these tutorials. Knowing the basic terms such as classpath, virtual machine, "strongly-typed" languages, and functional programming will help you debug, as well as expand on the knowledge you gain here. If you don't know Scala and want to learn it, Coursera has a great course named [Functional Programming Principles in Scala](https://www.coursera.org/learn/progfun1).

### Install Apache Zeppelin

#### Run via Docker

[Docker](https://www.docker.com/) is an easy-to-use containerization platform. This is the preferred method for running Zeppelin. Download the latest release from the [Skymind Docker Hub](https://hub.docker.com/r/skymindops/zeppelin-dl4j/).

We've assembled a special Docker with all dependencies installed:
```
docker run -it --rm  -p 8080:8080 skymindops/zeppelin-dl4j:latest
```

If you have a CUDA-enabled GPU and have [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) installed:
```
nvidia-docker run -it --rm  -p 8080:8080 skymindops/zeppelin-dl4j:latest-cuda-8.0
```

#### Via Binaries

Native binaries are also available for Zeppelin, downloadable here: https://zeppelin.apache.org/download.html.

## Setting up dependencies

If your installation of Zeppelin is not already set up for Deeplearning4j (i.e.: you didn't use our custom Docker image), you will need to add DL4J to the classpath. The easiest solution is to add the appropriate Maven dependencies to the included Spark Interpreter.

See this Zeppelin documentation for accessing the interpreter settings: https://zeppelin.apache.org/docs/latest/manual/dependencymanagement.html.

Once you have located the Spark Interpreter, you will need to add the following Maven library references:

| artifact | exlude | when to use? |
|---|---|---|
| `org.nd4j:nd4j-native-platform:0.9.1` | n/a | CPU-only machines |
| `org.nd4j:nd4j-cuda-8.0-platform:0.9.1` | n/a | GPU-enabled machines w/ CUDA |
| `org.deeplearning4j:deeplearning4j-core:0.9.1` | n/a | CPU-only or GPU machines w/o CuDNN |
| `org.deeplearning4j:deeplearning4j-cuda-8.0:0.9.1` | n/a | GPU machines w/ CuDNN installed |
| `org.deeplearning4j:deeplearning4j-zoo:0.9.1` | n/a | native zoo functionality (pretrained models) |
| `org.datavec:datavec-spark_2.11:0.9.1_spark_2` | `org.scala-lang:scala-compiler` | always |
| `org.deeplearning4j:dl4j-spark_2.11:0.9.1_spark_2` | `org.scala-lang:scala-compiler` | always |


Alternatively, you can dynamically load dependencies into notebooks, though this is not recommended. If you intend on adding new dependencies, you will have to restart the interpreter before re-running dynamic loading code. With that said, here's an example on how to do it:

```
%spark.dep

// if you are running Zeppelin for the first time, use this code block to load dependencies (see README above)
// note that if Zeppelin's spark interpreter has already been run, you will need to restart the interpreter
// clean up any previously loaded dependencies
z.reset()

// now load ND4J for CPU, our native tensor computing library
z.load("org.nd4j:nd4j-native-platform:0.9.1")

// or if you have a CUDA-enabled GPU, you can load ND4J for CUDA
// z.load("org.nd4j:nd4j-cuda-8.0-platform:0.9.1")

// finally, load the core deeplearning4j library with all basic features
z.load("org.deeplearning4j:deeplearning4j-core:0.9.1")

// don't forget to type Shift-Enter to run!
```

## Out-of-memory

Zeppelin may run out of memory when using larger networks. Its default memory setting is low. To fix this, create a `zeppelin-env.sh` file [like this one](https://github.com/apache/zeppelin/blob/master/conf/zeppelin-env.sh.template#L23) and enable the `ZEPPELIN_INTP_MEM` option.

```
export ZEPPELIN_INTP_MEM="-Xmx10g"
```

Increase the `-Xmx` option to something higher than 5GB of RAM. If you plan on using complex convolutional networks like VGG-16, you may need `-Xmx18g` or higher.

## Importing notebooks

Once your Zeppelin environment is set up, you can start importing our tutorials (if they aren't already included in a Docker image). Load the Zeppelin UI using the default host and port (likely http://localhost:8080/) and you should see "Welcome to Zeppelin!" on your screen. Once you have loaded this page, the `Import Note` link will be just below the Notebook column header.

**Note:** The notebooks use Zeppelin's JSON format. The `*.ipynb` format is for users who want to view the notebook using nbviewer or natively in Github.
