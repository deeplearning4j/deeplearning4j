---
title:
layout: default
---

Quick Start Guide
=========================================

## Prerequisites

This Quick Start guide assumes that you have the following already installed:

1. Java 7
2. IntelliJ (or another IDE)
3. Maven (Automated build tool)
4. Github 
 
If you need to install any of the above, please read how in the [ND4J Getting Started guide](http://nd4j.org/getstarted.html).  (ND4J is the linear-algebra engine behind Deeplearning4j, and the instructions there apply to both projects.) Don't install everything listed on the page, just the software listed above. 

We highly recommend you join our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) if you have questions or feedback. Even if you're feeling anti-social or brashly independent, feel free to lurk and learn.

## DL4J in a Few Easy Steps

After those installs, if you can follow these steps, you'll be up and running (Windows users please see the [Walkthrough](#walk) section below):

* Enter `git clone https://github.com/deeplearning4j/dl4j-0.4-examples` in your command line. (We are currently on examples version 0.0.4.x.)
* Import a new project in IntelliJ with Maven. 
* Copy and paste to make sure your POM.xml file looks like [this](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml). 
* For an easier install, Windows users should replace *nd4j-jblas* with *nd4j-java* in the dependencies. 
* Select `DBNIrisExample.java` from the lefthand file tree.
* Hit run! (It's the green button.)

You should get an F1 score of about 0.66, which is good for a small dataset like Iris. For a line by line walkthrough of the example, please refer to our [Iris DBN tutorial](../iris-flower-dataset-tutorial.html).

## Dependencies and Backends

Backends are what power the linear algebra operations behind DL4J's neural nets. Backends vary by chip. CPUs work fastest with both Jblas and Netlib Blas; GPUs with Jcublas. You can find all backends on [Maven Central](https://search.maven.org); click the linked version number under "Latest Version"; copy the dependency code on the left side of the subsequent screen; and paste it into your project root's pom.xml in IntelliJ. 

The *nd4j-java* backend will look something like this:

     <dependency>
       <groupId>org.nd4j</groupId>
       <artifactId>nd4j-java</artifactId>
       <version>${nd4j.version}</version>
     </dependency>

*nd4j-java* doesn't require Blas, which makes for the easiest setup on Windows. It works on all examples with DBNs, or deep-belief nets, but not on the other examples. 

The nd4j-jblas backend will look something like this:

     <dependency>
       <groupId>org.nd4j</groupId>
       <artifactId>nd4j-jblas</artifactId>
       <version>${nd4j.version}</version>
     </dependency>

*nd4j-jblas* works with all examples. To install Jblas, Windows users should refer to the [Deepelearining4j Getting Started page](../gettingstarted.html).

## Advanced: Using the Command Line on AWS

If you install Deeplearning4j on an AWS server with a Linux OS, you may want to use the command line to run your first examples, rather than relying on an IDE. In that case, run the *git clone*s and *mvn clean install*s according to the instruction above. With the installs completed, you can run an actual example with one line of code in the command line. The line will vary depending on the repo version and the specific example you choose. 

Here is a template:

    java -cp target/nameofjar.jar fully.qualified.class.name

And here is a concrete example, to show you roughly what your command should look like:

    java -cp target/dl4j-0.4-examples.jar org.deeplearning4j.MLPBackpropIrisExample

That is, there are two wild cards that will change as we update and you go through the examples:

    java -cp target/*.jar org.deeplearning4j.*

To make changes to the examples from the command line and run that changed file, you could, for example, tweak *MLPBackpropIrisExample* in *src/main/java/org/deeplearning4j/multilayer* and then maven-build the examples again. 

## Scala 

A [Scala version of the examples is here](https://github.com/kogecoo/dl4j-0.4-examples-scala).

## Next Steps

Once you've run the examples, please visit our [Getting Started page](../gettingstarted.html) to explore further. 

## <a name="walk">Walkthrough</a>

* Type the following into your command line to see if you have Git.

		git --version 

* If you do not, install [git](https://git-scm.herokuapp.com/book/en/v2/Getting-Started-Installing-Git). 
* In addition, set up a [Github account](https://github.com/join) and download GitHub for [Mac](https://mac.github.com/) or [Windows](https://windows.github.com/). 

* For Windows, find "Git Bash" in your Startup Menu and open it. The Git Bash terminal should look like cmd.exe.

* `cd` into the directory where you want to place the DL4J examples. You may want to create a new one with `mkdir ddl4j-examples` and then `cd` into that. Then run:

    git clone https://github.com/deeplearning4j/dl4j-0.4-examples

* Make sure the files were downloaded by entering `ls`. 
* Now open IntelliJ. 
* Click on the "File" menu, and then on "Import Project" or "New Project from Existing Sources". This will give you a local file menu. 
* Select the directory that contains the DL4J examples. 
* In the next window, you will be presented with a choice of build tools. Select Maven. 
* Check the boxes for "Search for projects recursively" and "Import Maven projects automatically" and click "Next." 
* Make sure your JDK/SDK is set up, and if it's not, click on the plus sign at the bottom of the SDK window to add it. 
* Then click through until you are asked to name the project. The default project name should do, so hit "Finish".
