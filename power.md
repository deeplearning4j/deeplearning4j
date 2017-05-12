---
title: Deeplearning4j on Power
layout: default
---

Deep Learning on Power Chips
----------------------

The [Power architecture](https://en.wikipedia.org/wiki/POWER8) is a widely used hardware architecture
shepherded by IBM. Suitable for high end servers, it's a great chip architecture
for doing deep learning. With the recent addition of [nvlink](http://www.nvidia.com/object/nvlink.html)
it's quickly becoming the go-to CPU architecture for deep learning applications.

Deeplearning4j can run on Power with zero code changes by including an [nd4j backend](http://nd4j.org/backend.html)
called [nd4j-native-platform](http://repo1.maven.org/maven2/org/nd4j/nd4j-native-platform/). 
Declare the latest version listed in your POM.xml file as you would with any other normal JVM-based project.

Why Maven or (Gradle, SBT,..)
-------------------------------

Our instructions here cover the Maven, but many of the terms for all automated build tools are similar. An [example presentation](http://www.slideshare.net/fabiofumarola1/3-maven-gradle-and-sbt) can be found here comparing the three listed above. 

We use an automated build tool rather than an OS-level package manager because Java is platform neutral. This is both good and bad. Maven and the associated tooling have their own storage, called Maven Central, which handles distribution of dependencies. Java IDEs are very well integrated with these tools. Linux-based package managers do not tend to map well to Java dependencies because of the sheer number of those dependencies.

If you are trying to build an application that is going to run on a Power server, we recommend using the uber JAR approach instead. One thing that might make things easier is using an uber JAR as part of an RPM or DEB package. This decouples deployment from application development.

Other Examples
----------------------

We have similar instructions for running on [GPUs](https://deeplearning4j.org/gpu) and [Android](https://deeplearning4j.org/android).

Running any of our [examples](https://github.com/deeplearning4j/dl4j-examples) should work out of the box. This is due to the fact `nd4j-native-platform` bundles all native dependencies (including Power). For more information running our examples, please see our [Quickstart](http://deeplearning4j.org/quickstart).

For running on a server, it is fairly easy to use Maven to create an [uber JAR](http://stackoverflow.com/questions/11947037/what-is-an-uber-jar).

In our examples, we use the [Maven Shade plugin](https://maven.apache.org/plugins/maven-shade-plugin/) for assembling all possible needed dependencies in to one jar. You can see where we do this on our examples [here](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/pom.xml#L140).

If you have any problems using Deeplearning4j on Power, please come to our [Gitter channel](https://gitter.im/deeplearning4j/deeplearning4j).
