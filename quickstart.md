---
title:
layout: default
---

#Quickstart

1. If you don't have Java 7 installed on your machine, download the [Java Development Kit (JDK) here](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html). The download will vary by operating system. For newer Macs, you'll want the file on this line:

		Mac OS X x64 185.94 MB -  jdk-7u60-macosx-x64.dmg

  You can test which version of Java you have (and whether you have it at all), by typing the following into the command line:

		java -version

2. Due to our reliance on Jblas for CPUs, native bindings for Blas are required.

		Fedora/RHEL
		yum -y install blas

		Ubuntu
		apt-get install libblas* (credit to @sujitpal)

		OSX
		Already Installed

		Windows
		See http://icl.cs.utk.edu/lapack-for-windows/lapack/

3. Since DL4J uses cross-platform tooling to make Python calls for data visualization and debugging, you'll also want to dowload [Anaconda here](http://continuum.io/downloads).

  Once you have Anaconda installed, you can test whether you have the necessary libs by entering this in a Python window:

		import numpy
		import pylab as pl

  These will generate the visualizations that allow you to debug your neural nets as they train. 

4. Next, download [DL4J examples here](https://oss.sonatype.org/content/repositories/snapshots/org/deeplearning4j/deeplearning4j-examples/0.0.3.2-SNAPSHOT/). Just click on the latest tar.gz or .zip file somewhere near the top of the list. It will look something like this:

		WINDOWS
		deeplearning4j-examples-0.0.3.2-20140811.044400-46-bin.zip

		MAC
		deeplearning4j-examples-0.0.3.2-20140811.044400-46-bin.tar.gz

Unzip the file. (*N.B.: If you have previously installed DL4J following the instructions on the Getting Started page, you already have this file.*)

To run the examples, move into the DL4J examples folder you downloaded and unzipped to make it your current working directory. To do that, you'll type something like this (file paths may vary :)

		cd Desktop/deeplearning4j-examples-0.0.3.2-SNAPSHOT
	
(*If you previously installed DL4J*, cd into */deeplearning4j-examples/target/, and there you will find the file you need to unzip.)

Once you've made the examples folder your current working directory, enter this command:

		java -cp "lib/*" org.deeplearning4j.example.mnist.RBMMnistExample

Now you should see evidence in your terminal/cmd that the neural net has begun to train. Look at the second-to-last number on the right hand side below. It should be decreasing. That’s the measure of the net’s error reconstructing a numeral-image. If the error shrinks, that means your net is learning, and all is well in the world.

![Alt text](../img/learning.png)

At the end of the training, you should see some numeral-images pop up in small windows in the upper left hand corner of your screen. Those are the reconstructions that prove your net actually works, and they look similar to these.

![Alt text](../img/two.png)![Alt text](../img/nine.png)![Alt text](../img/three.png)![Alt text](../img/one.png)

By this point, you should have trained your first neural net. Congratulations. (If you haven't, please [let us know](groups.google.com/forum/#!forum/deeplearning4j)!)

Go [here](https://github.com/agibsonccc/java-deeplearning/tree/master/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/) to choose the next example to run. Here is a tutorial on [how to run any example on DL4J](../runexample.html). 

Now it's time to start thinking about how to train it on other data. Check out the repos on our [Getting Started page](../gettingstarted.html).
