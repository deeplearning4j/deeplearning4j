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
called (nd4j-native-platform(http://repo1.maven.org/maven2/org/nd4j/nd4j-native-platform/). 
Declare the latest version listed in your pom.xml as any other normal JVM based project.


Running any of our [examples(https://github.com/deeplearning4j/dl4j-examples) should work out of the box.
For more information running our examples, please see our [quickstart](http://deeplearning4j.org/quickstart)
If you have any problems with using dl4j on power, please come to our [gitter channel](https://gitter.im/deeplearning4j/deeplearning4j)
