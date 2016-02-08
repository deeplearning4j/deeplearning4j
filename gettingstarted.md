---
title: Full Installation of Deeplearning4j
layout: default
---

# Full Installation

This is a multistep install. We highly recommend you join our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) if you have questions or feedback, so we can walk you through it. If you're feeling anti-social or brashly independent, you're still invited to lurk and learn. In addition, if you are utterly new to deep learning, we've got [a road map of what to learn when you're starting out](../deeplearningforbeginners.html). 

To run our examples in a few steps, please go to the [quickstart page](../quickstart.html) now. Seriously, go do that before you do this. It's the easy way to start with DL4J. 

The prerequisite installs for Deeplearning4j are documented on the [ND4J "Getting Started" page](http://nd4j.org/getstarted.html) of ND4J, the linear algebra engine powering DL4J's neural nets:

1. [Java 7 or above](http://nd4j.org/getstarted.html#java) 
2. [Integrated Development Environment: IntelliJ](http://nd4j.org/getstarted.html#ide-for-java) 
3. [Maven](http://nd4j.org/getstarted.html#maven)

After those installs, please read the following:

6. OS-specific instructions
    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
8. [GitHub](http://nd4j.org/getstarted.html#github)
9. <a href="#eclipse">Eclipse</a>
10. <a href="#trouble">Troubleshooting</a>
11. <a href="#results">Reproducible Results</a>
12. <a href="#next">Next Steps</a>

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

* Blas is already installed on OSX.  

### <a name="windows">Windows</a>

* While our Windows install is not always easy, Deeplearning4j is one of the few open-source deep learning projects that actually cares about trying to support the Windows community. Please see the [Windows section of our ND4J page](http://nd4j.org/getstarted.html#windows) for more instructions. 

* Install [MinGW 32 bits](http://www.mingw.org/) even if you have a 64-bit computer (the download button is on the upper right), and then download the [Prebuilt dynamic libraries using Mingw](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw). 

* Install [Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/). (Lapack will ask if you have Intel compilers. You do not.)

* Lapack offers the alternative of [VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke). You'll also want to look at the documentation for [Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/). 

* Alternatively, you can bypass MinGW and copy the Blas dll files to a folder in your PATH. For example, the path to the MinGW bin folder is: /usr/x86_64-w64-mingw32/sys-root/mingw/bin. To read more about the PATH variable in Windows, please read the [top answer on this StackOverflow page](https://stackoverflow.com/questions/3402214/windows-7-maven-2-install). 

* Cygwin is not supported. You must install DL4J from **DOS Windows**.  

* Running this file, [WindowsInfo.bat](https://gist.github.com/AlexDBlack/9f70c13726a3904a2100), can help debug your Windows install. Here's one [example of its output](https://gist.github.com/AlexDBlack/4a3995fea6dcd2105c5f) that shows what to expect. First download it, then open a command window / terminal. `cd` to the directory to which it was dowloaded. Enter `WindowsInfo` and hit enter. To copy its output, right click on command window -> select all -> hit enter. Output is then on clipboard.

For OpenBlas (see below) on **Windows**, download this [file](https://www.dropbox.com/s/6p8yn3fcf230rxy/ND4J_Win64_OpenBLAS-v0.2.14.zip?dl=1). Extract to somewhere such as `C:/BLAS`. Add that directory to your system's `PATH` environment variable.

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

###<a name="eclipse">Eclipse</a> 

After running a `git clone`, enter this command

      mvn eclipse:eclipse 
  
which will import the source and set everything up. 

After many years using Eclipse, we recommend IntelliJ, which has a similar interface. Eclipse's monolithic architecture has a tendency to cause strange errors in our code and others'. 

If you use Eclipse, you will need to install the [Lombok plugin](https://projectlombok.org/). You will also need the Maven plugin for Eclipse: [eclipse.org/m2e/](https://eclipse.org/m2e/).

Michael Depies has written this guide to [installing Deeplearning4j on Eclipse](https://depiesml.wordpress.com/2015/08/26/dl4j-gettingstarted/).

### <a name="trouble">Troubleshooting</a>

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

### <a name="next">Next Steps: IRIS Example & Building NNs</a>

In order to get started building neural nets, checkout the [Neural Nets Overview](http://deeplearning4j.org/neuralnet-overview.html) for more information.

Take a look at the [IRIS tutorial](../iris-flower-dataset-tutorial.html) to get running quickly, and check out our guide for [restricted Boltzmann machines](../restrictedboltzmannmachine.html) to understand the basic mechanics of *deep-belief networks*.

Follow the [ND4J Getting Started](http://nd4j.org/getstarted.html) instructions to start a new project and include necessary [POM dependencies](http://nd4j.org/dependencies.html). 
