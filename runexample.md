---
title:
layout: default
---

**prerequisites**: *you've [set up native matrices with Blas](../gettingstarted.html)*

#How to Run Any DL4J Example

Java involves deep and complex file systems, which you'll need to work with to run examples. 

The first thing you have to do is download a compressed file from [here](https://oss.sonatype.org/content/repositories/snapshots/org/deeplearning4j/deeplearning4j-examples/0.0.3.2-SNAPSHOT/). Windows users will probably prefer the zip file, while Mac users may opt for the tar.gz. Versions will vary over time, but the file names will look roughly like this:

		WINDOWS
		deeplearning4j-examples-0.0.3.2-20140811.044400-46-bin.zip
		
		MAC
		deeplearning4j-examples-0.0.3.2-20140811.044400-46-bin.tar.gz

Once you've unzipped your file, you'll need to cd into the SNAPSHOT folder, which will have a version-dependent name. Your command will look similar to this: 

		cd deeplearning4j-examples-0.0.3.2-SNAPSHOT

With the SNAPSHOT folder as your working directory, you can run any example it contains. The MNIST example is where most DL4J users start:

		java -cp "lib/*" org.deeplearning4j.example.mnist.RBMMnistExample

If windows pop up with graphs and numeral-image reconstructions, as they should if Numpy is working properly, you'll need to close them to keep the training going. The graphs' x-axis is narrow, but if they display normal (bell-curve-shaped) distributions, that indicates your net is training well.

###converting file paths to the command line

The jar files contained in your SNAPSHOT folder do not show the names of all the examples you can run. A list of those examples, however, can be found on [DL4J's GitHub repo](https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example). 

To translate GitHub file paths into file paths suitable for the command line, drop the first half of the GitHub URL, take the latter half starting at "org," and replace the remaining forward slashes with dots, as in the example below. And remember to drop the .java at the end.

For example, the file at 

[https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/convnet/mnist/MnistExampleMultiThreaded.java](https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/mnist/MnistExampleMultiThreaded.java)

can be translated to the command line as

		java -cp "lib/*" org.deeplearning4j.example.mnist.MnistExampleMultiThreaded

Here are a few other examples you can run, each of which takes about as long as the Mnist example above: 
		
		java -cp "lib/*" org.deeplearning4j.example.iris.IrisRBMExample
		
		java -cp "lib/*" org.deeplearning4j.example.lfw.MultiThreadedLFW
		

A fuller explanation of class paths in Java can be found in [Oracle's  documentation](http://docs.oracle.com/javase/8/docs/technotes/tools/windows/classpath.html).
