# ScalNet

ScalNet is a wrapper around Deeplearning4J emulating a [Keras](https://github.com/fchollet/keras) like API for deep learning.
 
ScalNet is released under an Apache 2.0 license. By contributing code to this repository, you agree to make your contribution available under an Apache 2.0 license.

ScalNet is STILL ALPHA and we are open sourcing this in an attempt to get feedback.

Come in to [gitter](https://gitter.im/deeplearning4j/deeplearning4j) if you are interested in learning more.


# Prerequisites

* JDK 8
* Scala 2.11.+ or 2.10.x
* SBT and Maven


# How to build

Currently ScalNet depends on deeplearning4j and nd4j SNAPSHOTS. 

- [deeplearning4j/deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)
- [deeplearning4j/nd4j](https://github.com/deeplearning4j/nd4j)

### sbt

ScalNet uses sbt, but due to [resolving issues](https://nd4j.org/dependencies), you must have Maven available to copy some nd4j-native dependencies in your classpath, in order to run the examples.

This is automatically done in `build.sbt` and you don't need to do anything besides having maven installed.

If you use sbt in your own project, you will probably have to proceed the same way. When ScalNet will be using releases instead of snapshots, this won't be necessary anymore.

To build, use:

```scala
$ sbt package
```

Alternatively, for some quick testing or usage in Jupyter for example, run:

```scala
$ sbt assembly
```
To obtain a JAR file with all needed dependencies.

See the [official sbt documentation](http://www.scala-sbt.org/documentation.html) for more on how to use sbt.

### Maven

Althought Maven is mainly used for release management, you can use the provided pom.xml to import ScalNet as a Maven project.

Target for scala 2.11

```scala
$ change-scala-versions.sh "2.11"
$ mvn package
```

Target for scala 2.10

```scala
$ change-scala-versions.sh "2.10"
$ mvn package
```

# How to use

### sbt

```scala
libraryDependencies ++= "org.deeplearning4j" % "scalnet_2.11" % "0.9.2-SNAPSHOT"
```

### Maven

```xml
<dependency>
    <groupId>org.deeplearning4j</groupId>
    <artifactId>scalnet_2.11</artifactId>
    <version>0.9.2-SNAPSHOT</version>
</dependency>
```


# Getting started

ScalNet uses a Keras like API, wrapping deeplearning4j to make it more easier to start with.
 
Also, since you can call Java code from Scala, you can still use everything from deeplearning4j. 

To see what ScalNet has to offer, run one of the [examples](https://github.com/deeplearning4j/ScalNet/tree/master/src/test/scala/org/deeplearning4j/scalnet/examples) it ships with.

Please note that those examples are not state-of-the-art in any way, they're just enough to get you started.
