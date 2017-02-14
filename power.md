---
title: Deeplearning4j on Power
layout: default
---

Deep Learning on Power 
----------------------

The [power architecture](https://en.wikipedia.org/wiki/POWER8) is a widely used hardware architecture
shepherded by IBM. Suitable for high end servers, it is a great chip architecture
for doing deep learning. With the recent addition of [nvlink](http://www.nvidia.com/object/nvlink.html)
it is quickly becoming the go to CPU architecture for deep learning applications.

Deeplearning4j can run on power with zero code changes by including an [nd4j backend](http://nd4j.org/backend.html)
called [nd4j-native-platform](http://repo1.maven.org/maven2/org/nd4j/nd4j-native-platform/). 
Declare the latest version listed in your pom.xml as any other normal JVM based project.

Why maven or (gradle,sbt,..)
-------------------------------
Our instructions here cover using maven but many of the terms for all automated build tools are similar, an [example presentation](http://www.slideshare.net/fabiofumarola1/3-maven-gradle-and-sbt) can be found here comparing the 3. 

We use an automated build tool rather than an OS level package manager. The reason for this is java being platform neutral.
This is both good and bad. Maven and the associated tooling have their own storage called maven central
which handles distribution of dependencies. Java ides are very well integrated with these tools. Linux based package
managers do not tend to map well to java dependencies because of the sheer number of them.

If you are trying to build an application that is going to run on a power server, We would recommend just using the uber jar approach instead. One thing that might make things easier is using an uber jar as part of an RPM or DEB package.
This decouples deployment from application development.

Other examples
----------------------
We have similar instructions for running on [gpus](https://deeplearning4j.org/gpu)
and [android](https://deeplearning4j.org/android)


Running any of our [examples](https://github.com/deeplearning4j/dl4j-examples) should work out of the box.
This is due to the fact nd4j-native-platform bundles all native dependencies (including power)
For more information running our examples, please see our [quickstart](http://deeplearning4j.org/quickstart)

For running on a server, it is fairly easy to use maven to create an [uber jar](http://stackoverflow.com/questions/11947037/what-is-an-uber-jar)

In our examples, we use the [maven shade plugin](https://maven.apache.org/plugins/maven-shade-plugin/)
for assembling all possible needed dependencies in to one jar. You can see where we do this on our examples
[here](https://github.com/deeplearning4j/dl4j-examples/blob/master/dl4j-examples/pom.xml#L140)


If you have any problems with using dl4j on power, please come to our [gitter channel](https://gitter.im/deeplearning4j/deeplearning4j)
