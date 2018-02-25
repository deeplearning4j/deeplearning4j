# ScalNet

ScalNet is a wrapper around Deeplearning4J emulating a [Keras](https://github.com/fchollet/keras) like API for deep learning.
 
ScalNet is released under an Apache 2.0 license. By contributing code to this repository, you agree to make your contribution available under an Apache 2.0 license.

ScalNet is STILL ALPHA and we are open sourcing this in an attempt to get feedback.

Come in to [gitter](https://gitter.im/deeplearning4j/deeplearning4j) if you are interested in learning more.


# Prerequisites

* JDK 8
* Scala 2.11.+
* SBT and Maven


# How to build

Currently ScalNet depends on deeplearning4j and nd4j SNAPSHOTS. 

- [deeplearning4j/deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)
- [deeplearning4j/nd4j](https://github.com/deeplearning4j/nd4j)

ScalNet use sbt, but due to [resolving issues](https://nd4j.org/dependencies), you must have maven available in order to copy some nd4j-native dependencies in classpath needed to run examples.

However, this is automatically done in `build.sbt` and you don't need to do anything besides having maven installed.


# How to use

In your own sbt project, add the dependency:

```scala
libraryDependencies ++= "org.deeplearning4j" % "scalnet_2.11" % "0.9.2-SNAPSHOT"
```

See the [official sbt documentation](http://www.scala-sbt.org/documentation.html) for more on how to use sbt.

Alternatively for some quick testing or usage in Jupyter for example, clone the ScalNet Github repository and run:

```scala
$ sbt assembly
```

To obtain a JAR file with all needed dependencies.


# Getting started

ScalNet use a Keras like API, wrapping deeplearning4j to make it more easier to start with.
 
Also, since you can use Java code from Scala, you can still use everything from deeplearning4j. 

To see what ScalNet has to offer, run one of the [examples](https://github.com/deeplearning4j/ScalNet/tree/master/src/test/scala/org/deeplearning4j/scalnet/examples) it ships with.
