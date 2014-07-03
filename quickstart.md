---
title:
layout: default
---

**prerequisites**

*you've gone through [these steps](../gettingstarted.html) and set up Maven and Blas (native matrices).*

#quickstart

If you don't have Java 7 installed on your machine, download the [Java Development Kit (JDK) here](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html). The download will vary by operating system. For newer Macs, you'll want the file on this line:

	Mac OS X x64 185.94 MB -  jdk-7u60-macosx-x64.dmg

You can test which version of Java you have (and whether you have it at all), by typing 

	java -version

in the command line.

Since DL4J uses cross-platform tooling to make Python calls for data visualization and debugging, you'll also want to dowload [Anaconda here](http://continuum.io/downloads).

Once you have Anaconda installed, you can test whether you have the necessary libs by entering this in the command line:

	import numpy
	import pylab as pl

These will generate the visualizations that allow you to debug your neural nets as they train. 

Next, download [DL4J examples here](https://oss.sonatype.org/content/repositories/snapshots/org/deeplearning4j/deeplearning4j-examples/0.0.3.2-SNAPSHOT/). Just click on the latest tar.gz file somewhere near the top of the list. It will look something like this:

	deeplearning4j-examples-0.0.3.2-20140625.144826-39-bin.tar.gz

Unzip the file.

To run the examples, move into the DL4J examples folder you downloaded and unzipped to make it your current working directory. To do that, you'll type something like this

		cd Desktop/deeplearning4j-examples-0.0.3.2-SNAPSHOT

Once you've made the the examples folder your current working directory, enter this command:

		java -cp "lib/*" org.deeplearning4j.example.mnist.RBMMnistExample

Now you should see evidence that the neural net has begun to train. Look at the second-to-last number on the right hand side below. It should be decreasing. That’s the measure of the net’s error reconstructing a numeral image. If the error shrinks, that means your net is learning, and all is well in the world.

![Alt text](../img/learning.png)

At the end of the training, you should see some numerals pop up in small windows in the upper left hand corner of your screen. Those are the reconstructions that prove your net works, and they look similar to these.

![Alt text](../img/two.png)![Alt text](../img/nine.png)![Alt text](../img/three.png)![Alt text](../img/one.png)

Go [here](https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/) to choose the next example to run.

Java works with deep and complex file systems, which are important to explain here. You can choose the example to run by dropping the first half of the URL and replacing the remaining forward slashes with dots, as in the example below.

For example, the file at 

[https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/convnet/mnist/MnistConvNetTest.java](https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/convnet/mnist/MnistConvNetTest.java)

can be entered into the command line as 

		java -cp "lib/*" org.deeplearning4j.example.mnist.MnistExampleMultiThreaded.java
		
Here are a few other examples you can run, each of which takes about as long as the Mnist example above:
		
		java -cp "lib/*" org.deeplearning4j.example.iris.IrisRBMExample
		
		java -cp "lib/*" org.deeplearning4j.example.lfw.MultiThreadedLFW

A fuller explanation of class paths in Java can be found in [Oracle's  documentation](http://docs.oracle.com/javase/8/docs/technotes/tools/windows/classpath.html).

By this point, you should have trained your first neural net. Congratulations. (If you haven't, please [let us know](groups.google.com/forum/#!forum/deeplearning4j)!)

Now it's time to start thinking about how to train it on other data. Check out the repos on our [Getting Started page](../gettingstarted.html) and then start exploring how to deal with [your own datasets](../customdatasets.html).
