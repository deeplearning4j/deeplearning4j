---
title:
layout: default
---

**prerequisites**: *you've gone through [these steps](../gettingstarted.html) and set up Maven and Blas (native matrices).*

#How to Run Any DL4J Example

Java works with deep and complex file systems, which are important to explain here. You can choose the example to run by dropping the first half of the URL, take the latter half starting at "org," and replace the remaining forward slashes with dots, as in the example below. For example, the file at 

[https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/convnet/mnist/MnistExampleMultiThreaded.java](https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/mnist/MnistExampleMultiThreaded.java)

can be entered into the command line as 

		java -cp "lib/*" org.deeplearning4j.example.mnist.MnistExampleMultiThreaded.java
		
Here are a few other examples you can run, each of which takes about as long as the Mnist example above:
		
		java -cp "lib/*" org.deeplearning4j.example.iris.IrisRBMExample.java
		
		java -cp "lib/*" org.deeplearning4j.example.lfw.MultiThreadedLFW.java

A fuller explanation of class paths in Java can be found in [Oracle's  documentation](http://docs.oracle.com/javase/8/docs/technotes/tools/windows/classpath.html).

