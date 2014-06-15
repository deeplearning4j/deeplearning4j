---
title:
layout: default
---

# Pre reqs

We assume you have already gone through the [setup](http://deeplearning4j.org/gettingstarted.html) This assumes you have setup maven, blas (native matrices)


# quickstart



1. Download these [DL4J examples](https://oss.sonatype.org/content/repositories/snapshots/org/deeplearning4j/deeplearning4j-examples/0.0.3.2-SNAPSHOT/).

2. Unzip file.

3. Run examples

		cd [root of path to distribution]

		java -cp "lib/*" org.deeplearning4j.example.mnist.RBMMnistExample

4. Go [here](https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/) to choose the next example to run.

Java works with deep and complex file systems, which are important to explain here. You can choose the example to run by replacing the forward slashes with dots.

For example, the file at 

[https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/convnet/mnist/MnistConvNetTest.java](https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/convnet/mnist/MnistConvNetTest.java)

can be entered into the command line as 

		java -cp "lib/*" org.deeplearning4j.example.convnet.mnist.MnistConvNetTest

A fuller explanation of class paths in Java can be found [in Oracle's documentation](http://docs.oracle.com/javase/8/docs/technotes/tools/windows/classpath.html).
