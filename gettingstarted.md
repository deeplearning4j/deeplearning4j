---
title: Full Installation of Deeplearning4j
layout: default
---

# DL4J Comprehensive Setup Guide

This page builds on the instructions in the [Quick Start Guide](http://deeplearning4j.org/quickstart), and provides additional details and some troubleshooting steps. Seriously, go and read that page first before you proceed with this. It's the easy way to start with DL4J.

This is a multistep install. We highly recommend you join our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) if you have questions or feedback, so we can walk you through it. If you're feeling anti-social or brashly independent, you're still invited to lurk and learn. In addition, if you are utterly new to deep learning, we've got [a road map of what to learn when you're starting out](../deeplearningforbeginners.html). 


After following the steps in the [Quick Start Guide](http://deeplearning4j.org/quickstart), please read the following:

1. Accelerating CPU Training: Installing Native BLAS Libraries
    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
2. [GitHub](http://nd4j.org/getstarted.html#github)
3. <a href="#eclipse">Eclipse</a>
4. <a href="#cli">Command-Line Interface</a>
5. <a href="#trouble">Troubleshooting</a>
6. <a href="#results">Reproducible Results</a>
7. <a href="#next">Next Steps</a>


## Accelerating CPU Training Performance: Installing Native BLAS Libraries

Neural network training is computationally expensive. In order to obtain high computational performance, ND4J makes use of native (c/fortran) basic linear algebra subpgropgams (BLAS) libraries, such as [OpenBLAS](http://www.openblas.net/). Though ND4J will operate correctly without it, training performance in DL4J will suffer.


### <a name="linux">Linux</a>

* Due to our reliance on various forms of Blas for CPUs, native bindings for Blas are required.

        Fedora/RHEL
        yum -y install blas

        Ubuntu
        apt-get install libblas* (credit to @sujitpal)

Please see [this section](#open) for more information on OpenBlas.

* If GPUs are broken, you'll need to enter an extra command. First, find out where Cuda installs itself. It will look something like this

         /usr/local/cuda/lib64

Then enter *ldconfig* in the terminal, followed by the file path to link Cuda. Your command will look similar to this

         ldconfig /usr/local/cuda/lib64

If you're still unable to load Jcublas, you will need to add the parameter -D to your code (it's a JVM argument):

         java.library.path (settable via -Djava.librarypath=...) 
         // ^ for a writable directory, then 
         -D appended directly to "<OTHER ARGS>" 

If you're using IntelliJ as your IDE, this should work already. 

### <a name="osx">OSX</a>

* OSX ships with a fast native BLAS library, [vecLib](https://developer.apple.com/library/mac/documentation/Performance/Conceptual/vecLib/). In most cases, this is sufficient for users, and is quite fast.

### <a name="windows">Windows</a>

To install native binaries (OpenBLAS) on Windows, you have two options.

- Option 1: Use a pre-compiled version of OpenBLAS, as described below (recommended)
- Option 2: Compile OpenBLAS from source. This may result in faster execution due to machine-specific optimizations possible during compilation, but is *considerably* more complicated than using the precompiled binaries.

For the *precompiled* version of OpenBlas (see below) on **Windows**, download [this file](https://www.dropbox.com/s/6p8yn3fcf230rxy/ND4J_Win64_OpenBLAS-v0.2.14.zip?dl=1). Extract to somewhere such as `C:/BLAS`. Add that directory to your system's `PATH` environment variable, and then restart your IDE (and, ideally your computer too). If that doesn't work, please refer to this [Github issue](https://github.com/deeplearning4j/deeplearning4j/issues/1168).

To build OpenBLAS from source on Windows, see [this link](https://gist.github.com/AlexDBlack/c34e95fd08c0e9b3891b).

### <a id="open"> OpenBlas </a>

To make sure the native libs on the x86 backend work, you need `/opt/OpenBLAS/lib` on the system path. After that, enter these commands in the prompt

			sudo cp libopenblas.so liblapack.so.3
			sudo cp libopenblas.so libblas.so.3

We added this so that [Spark](http://deeplearning4j.org/spark) would work with OpenBlas.

If OpenBlas is not working correctly, follow these steps:

* Remove Openblas if you installed it.
* Run `sudo apt-get remove libopenblas-base`
* Download the development version of OpenBLAS
* `git clone git://github.com/xianyi/OpenBLAS`
* `cd OpenBLAS`
* `make FC=gfortran`
* `sudo make PREFIX=/usr/local/ install`
* With **Linux**, double check if the symlinks for `libblas.so.3` and `liblapack.so.3` are present anywhere in your `LD_LIBRARY_PATH`. If they aren't, add the links to `/usr/lib`. A symlink is a "symbolic link." You can set it up like this (the -s makes the link symbolic):

		ln -s TARGET LINK_NAME
		// interpretation: ln -s "to-here" <- "from-here"

* The "from-here" is the symbolic link that does not exist yet, and which you are creating. Here's StackOverflow on [how to create a symlink](https://stackoverflow.com/questions/1951742/how-to-symlink-a-file-in-linux). And here's the [Linux man page](http://linux.die.net/man/1/ln).
* As a last step, restart your IDE. 
* For complete instructions on how to get native Blas running with **Centos 6**, [see this page](https://gist.github.com/jarutis/912e2a4693accee42a94) or [this](https://gist.github.com/sato-cloudian/a42892c4235e82c27d0d).

For OpenBlas on **Ubuntu** (15.10), please see [these instructions](http://pastebin.com/F0Rv2uEk).

## <a name="walk">DL4J Examples: A More Details Step-by-Step Walkthrough</a>

This section provides a more comprehensive version of the steps contained in the [quickstart guide](http://deeplearning4j.org/quickstart).

* Type the following into your command line to see if you have Git.

		git --version 

* If you do not, install [git](https://git-scm.herokuapp.com/book/en/v2/Getting-Started-Installing-Git). 
* In addition, set up a [Github account](https://github.com/join) and download GitHub for [Mac](https://mac.github.com/) or [Windows](https://windows.github.com/). 
* For Windows, find "Git Bash" in your Startup Menu and open it. The Git Bash terminal should look like cmd.exe.
* `cd` into the directory where you want to place the DL4J examples. You may want to create a new one with `mkdir dl4j-examples` and then `cd` into that. Then run:

    `git clone https://github.com/deeplearning4j/dl4j-0.4-examples`
* Make sure the files were downloaded by entering `ls`. 
* Now open IntelliJ. 
* Click on the "File" menu, and then on "Import Project" or "New Project from Existing Sources". This will give you a local file menu. 
* Select the directory that contains the DL4J examples. 
* In the next window, you will be presented with a choice of build tools. Select Maven. 
* Check the boxes for "Search for projects recursively" and "Import Maven projects automatically" and click "Next." 
* Make sure your JDK/SDK is set up, and if it's not, click on the plus sign at the bottom of the SDK window to add it. 
* Then click through until you are asked to name the project. The default project name should do, so hit "Finish".

## <a name="eclipse">Using DL4J Examples in Eclipse</a> 

In IntelliJ, it is simply sufficient to import the examples as described in the quickstart guide. In order to use the example in Eclipse, an additional step is required. 

After running a `git clone`, run the following command in your command line:

      mvn eclipse:eclipse 
  
This will create an Eclipse project that you can then import.

After many years using Eclipse, we recommend IntelliJ, which has a similar interface. Eclipse's monolithic architecture has a tendency to cause strange errors in our code and others'. 

If you use Eclipse, you will need to install the Maven plugin for Eclipse: [eclipse.org/m2e/](https://eclipse.org/m2e/).

Michael Depies has written this guide to [installing Deeplearning4j on Eclipse](https://depiesml.wordpress.com/2015/08/26/dl4j-gettingstarted/).

## <a name="cli">Command-Line Interface</a>

`deeplearning4j-cli` can now be installed these two ways:

On Centos/Redhat, you can do:

		# install
		sudo yum install https://s3-us-west-2.amazonaws.com/skymind/bin/deeplearning4j-cli.rpm
		# run
		dl4j

On non-rpm systems, do:

		# download
		curl -O https://s3-us-west-2.amazonaws.com/skymind/bin/deeplearning4j-cli.tar.gz
		# untar
		tar -zxvf deeplearning4j-cli.tar.gz
		# run
		cd deeplearning4j-cli-0.4-rc3.9-SNAPSHOT ; ./bin/dl4j

## <a name="trouble">Troubleshooting</a>

* Please feel free to ask us about error messages on our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j). When you post your question, please have the following information ready (it will really speed things up!):

      * Operating System (Windows, OSX, Linux) and version 
      * Java version (7, 8) : type java -version in your terminal/CMD
      * Maven version : type mvn --version in your terminal/CMD
      * Stacktrace: Please past the error code on Gist and share the link with us: [https://gist.github.com/](https://gist.github.com/)
* If you have installed DL4J before and now see the examples throwing errors, please update your libraries. With Maven, just update the versions in your POM.xml file to match the latest versions on [Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j). With source, you can run a `git clone` on [ND4J](http://nd4j.org/getstarted.html), Canova and DL4J and a `mvn clean install -Dskiptests=true -Dmaven.javadoc.skip=true` within all three directories, in that order.
* When you run an example, you may get a low [f1 score](../glossary.html#f1), which is the probability that the net's classification is accurate. In this case, a low f1 doesn't indicate poor performance, because the examples train on small data sets. We gave them small data sets so they would run quickly. Because small data sets are less representative than large ones, the results they produce will vary a great deal. For example, on the minuscule example data, our deep-belief net's f1 score currently varies between 0.32 and 1.0. 
* Deeplearning4j includes an **autocomplete function**. If you are unsure which commands are available, press any letter and a drop-down list like this will appear:
![Alt text](../img/dl4j_autocomplete.png)
* Here's the **Javadoc** for all [Deeplearning4j's classes and methods](http://deeplearning4j.org/doc/).
* As the code base grows, installing from source requires more memory. If you encounter a `Permgen error` during the DL4J build, you may need to add more **heap space**. To do that, you'll need to find and alter your hidden `.bash_profile` file, which adds environmental variables to bash. To see those variables, enter `env` in the command line. To add more heap space, enter this command in your console:
      echo "export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=512m"" > ~/.bash_profile
* Older versions of Maven, such as 3.0.4, are likely to throw exceptions like a NoSuchMethodError. This can be fixed by upgrading to the latest version of Maven, which is currently 3.3.x. To check your Maven version, enter `mvn -v` in the command line.
* After you install Maven, you may receive a message like this: `mvn is not recognised as an internal or external command, operable program or batch file.` That means you need Maven in your [PATH variable](https://www.java.com/en/download/help/path.xml), which you can change like any other environmental variable.  
* If you see the error `Invalid JDK version in profile 'java8-and-higher': Unbounded range: [1.8, for project com.github.jai-imageio:jai-imageio-core com.github.jai-imageio:jai-imageio-core:jar:1.3.0`, you may have a Maven issue. Please update to version 3.3.x.
* To compile some ND4J dependencies, you need to install some **dev tools** for C and C++. [Please see our ND4J guide](http://nd4j.org/getstarted.html#devtools).
* The include path for [Java CPP](https://github.com/bytedeco/javacpp) doesn't always work on **Windows**. One workaround is to take the the header files from the include directory of Visual Studio, and put them in the include directory of the Java Run-Time Environment (JRE), where Java is installed. This will affect files such as standardio.h. More information is available [here](http://nd4j.org/getstarted.html#windows). 
* Instructions on monitoring your GPUs are [here](http://nd4j.org/getstarted.html#gpu).
* One major reason to use Java is its pre-baked diagnostics in the **[JVisualVM](https://docs.oracle.com/javase/8/docs/technotes/tools/unix/jvisualvm.html)**. If you have Java installed, just enter `jvisualvm` in your command line and you'll get visuals on your CPU, Heap, PermGen, Classes and Threads. One useful view: Click on the `Sampler` tab on the upper right, and then select the CPU or Memory button for visuals. 
![Alt text](../img/jvisualvm.png)
* Some problems encountered using DL4J may be due to a lack of familiarity with the ideas and techniques of machine learning. We strongly encourage all Deeplearning4j users to rely on resources beyond this website to understand the fundamentals. We've included a list of educational resources for machine and deep learning on [this page](../deeplearningpapers.html). While we've partially documented DL4J, parts of the code essentially remain a raw, domain-specific language for deep learning.
* When using `deeplearning4j-nlp` from a **Clojure** application and building an uberjar with Leiningen, it is necessary to specify the following in the `project.clj` so that the akka `reference.conf` resource files are properly merged. `:uberjar-merge-with {#"\.properties$" [slurp str spit] "reference.conf" [slurp str spit]}`. Note that the first entry in the map for .properties files is the usual default). If this is not done, the following exception will be thrown when trying to run from the resulting uberjar: `Exception in thread "main" com.typesafe.config.ConfigException$Missing: No configuration setting found for key 'akka.version'`.
* Float support is buggy on OSX. If you see NANs where you expect numbers running our examples, switch the data type to `double`.
* There is a bug in fork-join in Java 7. Updating to Java 8 fixes it. If you get an OutofMemory error that looks like this, fork join is the problem: `java.util.concurrent.ExecutionException: java.lang.OutOfMemoryError`
.... `java.util.concurrent.ForkJoinTask.getThrowableException(ForkJoinTask.java:536)`

### <a name="results">Reproducible Results</a>

Neural net weights are initialized randomly, which means the model begins learning from a different position in the weight space each time, which may lead it to different local optima. Users seeking reproducible results will need to use the same random weights, which they must initialize before the model is created. They can reinitialize with the same random weight with this line:

      Nd4j.getRandom().setSeed(123);
      
## Scala 

A [Scala version of the examples is here](https://github.com/kogecoo/dl4j-0.4-examples-scala).
      
### Managed Environments

If you are working in a managed environment like Databricks, Domino or Sense.io, you'll need to take an additional step. After you've followed the local setup above, just run 

		mvn clean package

in the command line from within the examples directory. Then you can upload the JAR file to the managed environment you've chosen.

## Advanced: Using the Command Line on AWS

If you install Deeplearning4j on an AWS server with a Linux OS, you may want to use the command line to run your first examples, rather than relying on an IDE. In that case, run the *git clone*s and *mvn clean install*s according to the instruction above. With the installs completed, you can run an actual example with one line of code in the command line. The line will vary depending on the repo version and the specific example you choose. 

Here is a template:

    java -cp target/nameofjar.jar fully.qualified.class.name

And here is a concrete example, to show you roughly what your command should look like:

    java -cp target/dl4j-0.4-examples.jar org.deeplearning4j.MLPBackpropIrisExample

That is, there are two wild cards that will change as we update and you go through the examples:

    java -cp target/*.jar org.deeplearning4j.*

To make changes to the examples from the command line and run that changed file, you could, for example, tweak *MLPBackpropIrisExample* in *src/main/java/org/deeplearning4j/multilayer* and then maven-build the examples again. 

### <a name="next">Next Steps: IRIS Example & Building NNs</a>

In order to get started building neural nets, checkout the [Neural Nets Overview](http://deeplearning4j.org/neuralnet-overview.html) for more information.

Take a look at the [IRIS tutorial](../iris-flower-dataset-tutorial.html) to get running quickly, and check out our guide for [restricted Boltzmann machines](../restrictedboltzmannmachine.html) to understand the basic mechanics of *deep-belief networks*.

Follow the [ND4J Getting Started](http://nd4j.org/getstarted.html) instructions to start a new project and include necessary [POM dependencies](http://nd4j.org/dependencies.html). 
