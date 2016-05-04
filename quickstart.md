---
title: Quick Start Guide for Deeplearning4j
layout: default
---
<!-- Begin Inspectlet Embed Code -->
<script type="text/javascript" id="inspectletjs">
window.__insp = window.__insp || [];
__insp.push(['wid', 1755897264]);
(function() {
function ldinsp(){if(typeof window.__inspld != "undefined") return; window.__inspld = 1; var insp = document.createElement('script'); insp.type = 'text/javascript'; insp.async = true; insp.id = "inspsync"; insp.src = ('https:' == document.location.protocol ? 'https' : 'http') + '://cdn.inspectlet.com/inspectlet.js'; var x = document.getElementsByTagName('script')[0]; x.parentNode.insertBefore(insp, x); };
setTimeout(ldinsp, 500); document.readyState != "complete" ? (window.attachEvent ? window.attachEvent('onload', ldinsp) : window.addEventListener('load', ldinsp, false)) : ldinsp();
})();
</script>
<!-- End Inspectlet Embed Code -->

Quick Start Guide
=================

This is everything you need to run DL4J examples and begin your own projects.

We recommend that you join our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j). Gitter is where you can request help and give feedback, but please do use this guide before asking questions we've answered below. If you are new to deep learning, we've included [a road map for beginners](../deeplearningforbeginners.html) with links to courses, readings and other resources. 

## Prerequisites

* [Java (developer version)](#Java) 1.7 or later
* [Apache Maven](#Maven)
* [IntelliJ IDEA](#IntelliJ) 
* [Git](#Git)

You *must* have all of these installed to use this Quick Start guide. DL4J targets professional Java developers who are familiar with production deployments, IDEs and automated build tools. Working with DL4J will be easiest if you already have experience with these.

If you are new to Java or unfamiliar with these tools, read the details below for help with installation and setup. Otherwise, **skip to <a href="#examples">DL4J Examples</a>**.

#### <a name="Java">Java</a>

If you don't have Java 1.7 or later, download the current [Java Development Kit (JDK) here](http://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html). To check if you have a compatible version of Java installed, use the following command:

``` shell
java -version
```

#### <a name="Maven">Apache Maven</a>

Maven is a dependency management and automated build tool for Java projects. It works well with IDEs such as IntelliJ and lets you install DL4J project libraries easily. [Install or update Maven](https://maven.apache.org/download.cgi) to the latest release following [their instructions](https://maven.apache.org/install.html) for your system. To check if you have the most recent version of Maven installed, enter the following:

``` shell
mvn --version
```

If you are working on a Mac, you can simply enter the following into the command line:

``` shell
brew install maven
```

Maven is widely used among Java developers and it's pretty much mandatory for working with DL4J. If you come from a different background and Maven is new to you, check out [Apache's Maven overview](http://maven.apache.org/what-is-maven.html) and our [introduction to Maven for non-Java programmers](http://deeplearning4j.org/maven.html), which includes some additional troubleshooting tips. 

#### <a name="IntelliJ">IntelliJ IDEA</a>

An Integrated Development Environment ([IDE](http://encyclopedia.thefreedictionary.com/integrated+development+environment)) allows you to work with our API and configure neural networks in a few steps. We strongly recommend using [IntelliJ](https://www.jetbrains.com/idea/download/), which communicates with Maven to handle dependencies. The [community edition of IntelliJ](https://www.jetbrains.com/idea/download/) is free. 

There are other popular IDEs such as [Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html) and [Netbeans](http://wiki.netbeans.org/MavenBestPractices). IntelliJ is preferred, and using it will make finding help on [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) easier if you need it.

#### <a name="Git">Git</a>

Install the [latest version of Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git). If you already have Git, you can update to the latest version using Git itself:

``` shell
$ git clone git://git.kernel.org/pub/scm/git/git.git
```

## <a name="examples">DL4J Examples in a Few Easy Steps</a>

1. Use command line to enter the following:

        $ git clone https://github.com/deeplearning4j/dl4j-0.4-examples.git
        $ cd dl4j-0.4-examples/
        $ mvn clean install

2. Open IntelliJ and choose Import Project. Then select the main 'dl4j-0.4-examples' directory. 
![select directory](../img/Install_IntJ_1.png)

3. Choose 'Import project from external model' and ensure that Maven is selected. 
![import project](../img/Install_IntJ_2.png)

4. Continue through the wizard's options. Select the SDK that begins with `jdk`. (You may need to click on a plus sign to see your options...) Then click Finish. Wait a moment for IntelliJ to download all the dependencies. You'll see the horizontal bar working on the lower right.

5. Pick an example from the file tree on the left.
![run IntelliJ example](../img/Install_IntJ_3.png)
Right-click the file to run. 

## Using DL4J In Your Own Projects: Configuring the POM.xml File

To run DL4J in your own projects, we highly recommend using Maven for Java users, or a tool such as SBT for Scala. The basic set of dependencies and their versions are shown below. This includes:

- `deeplearning4j-core`, which contains the neural network implementations
- `nd4j-x86`, the CPU version of the ND4J library that powers DL4J
- `canova-api` - Canova is our library vectorizing and loading data

Every Maven project has a POM file. Here is how the POM file should appear when you run your examples, and here is [a link to it on Github](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml).

<script src="http://gist-it.appspot.com/https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/pom.xml?slice=1:175"></script>

Within IntelliJ, you will need to choose the first Deeplearning4j example you're going to run. We suggest `MLPLinearClassifier`, as you will almost immediately see the network classify two groups of data in our UI. The file on [Github can be found here](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/feedforward/classification/MLPClassifierLinear.java). 

To run the example, right click on it and select the green button in the drop-down menu. You will see, in IntelliJ's bottom window, a series of scores. The rightmost number is the error score for the network's classifications. If your network is learning, then that number will decrease over time with each batch it processes. At the end, this window will tell you how accurate your neural-network model has become:

![run IntelliJ example](../img/mlp_classifier_results.png)

In another window, a graph will appear, showing you how the multilayer perceptron (MLP) has classified the data in the example. It will look like this:

![run IntelliJ example](../img/mlp_classifier_viz.png)

And congratulations! You just trained your first neural network with Deeplearning4j...

Additional links:

- [Deeplearning4j artifacts on Maven Central](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j)
- [ND4J artifacts on Maven Central](http://search.maven.org/#search%7Cga%7C1%7Cnd4j)
- [Canova artifacts on Maven Central](http://search.maven.org/#search%7Cga%7C1%7Ccanova)

## Next Steps

1. Join us on Gitter. We have three big community channels.
  * [DL4J Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) is the main channel for all things DL4J. Most people hang out here.
  * [Tunning Help](https://gitter.im/deeplearning4j/deeplearning4j/tuninghelp) is for people just getting started with neural networks. Beginners please visit us here!
  * [Early Adopters](https://gitter.im/deeplearning4j/deeplearning4j/earlyadopters) is for those who are helping us vet and improve the next release. WARNING: This is for more experienced folks. 
2. Read the [introduction to deep neural networks](../neuralnet-overview) or one of our detailed tutorials. 
3. Check out the more detailed [Comprehensive Setup Guide](../gettingstarted).
4. Browse the [DL4J documentation](../documentation).
