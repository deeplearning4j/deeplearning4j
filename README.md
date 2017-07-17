# ScalNet

ScalNet is a wrapper around Deeplearning4j emulating a [Keras](https://github.com/fchollet/keras) like API for deep learning. ScalNet is released under an Apache 2.0 license. By contributing code to this repository, you agree to make your contribution available under an Apache 2.0 license.

ScalNet is STILL ALPHA and we are open sourcing this in an attempt to get feedback.

You will have to build from source to use ScalNet (this includes the dl4j toolchain).

Come in to [gitter](https://gitter.im/deeplearning4j/deeplearning4j) if you are interested in learning more.

# How to build

Currently ScalNet depends on deeplearning4j and nd4j SNAPSHOTs. Please install these package in your local machine first with `mvn install`.

- [deeplearning4j/deeplearning4j](https://github.com/deeplearning4j/deeplearning4j)
- [deeplearning4j/nd4j](https://github.com/deeplearning4j/nd4j)

Target for scala-2.11

```scala
$ mvn package -Pscala-2.11.x
```

Target for scala-2.10

```scala
$ mvn clean -Pscala-2.10.x
```

Having built deeplearning4j and nd4j users can also `sbt` instead of `mvn` to build and test the project, e.g. using 
```$scala
$ sbt compile
```
to build the project, see the [official sbt documentation](http://www.scala-sbt.org/documentation.html) for more on how to use sbt.

# Getting started

To see what ScalNet has to offer, run one of the [examples](https://github.com/deeplearning4j/ScalNet/tree/master/src/test/scala/org/deeplearning4j/scalnet/examples) it ships with.
