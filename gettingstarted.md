---
layout: default
---

# Getting Started

This is a multistep install. We highly recommend you join our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j) if you have questions or feedback, so we can walk you through it. If you're feeling anti-social or brashly independent, you're still invited to lurk and learn. 

The prerequisites installs for Deeplearning4j are documented on the ["Getting Started" page](http://nd4j.org/getstarted.html) of ND4J, the linear algebra engine powering DL4J's neural nets:

1. [Java 7](http://nd4j.org/getstarted.html#java) 
2. [Integrated Development Environment: IntelliJ](http://nd4j.org/getstarted.html#ide-for-java) 
3. [Maven](http://nd4j.org/getstarted.html#maven)
4. [Canova: An ML Vectorization Lib](http://nd4j.org/getstarted.html#canova)
5. [GitHub](http://nd4j.org/getstarted.html#github)

After those installs, please read the following:

6. OS-specific instructions
    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
7. <a href="#source">Working With Source</a>
8. <a href="#eclipse">Eclipse</a>
9. <a href="#trouble">Troubleshooting</a>
10. <a href="#results">Reproducible Results</a>
11. <a href="#next">Next Steps</a>

### <a name="linux">Linux</a>

* Due to our reliance on Jblas for CPUs, native bindings for Blas are required.

        Fedora/RHEL
        yum -y install blas

        Ubuntu
        apt-get install libblas* (credit to @sujitpal)

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

* Jblas is already installed on OSX.  

### <a name="windows">Windows</a>

* Install [MinGW 32 bits](http://www.mingw.org/) even if you have a 64-bit computer (the download button is on the upper right), and then download the [Prebuilt dynamic libraries using Mingw](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw). 

* Install [Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/). (Lapack will ask if you have Intel compilers. You do not.)

* Lapack offers the alternative of [VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke). You'll also want to look at the documentation for [Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/). 

* Alternatively, you can bypass MinGW and copy the Blas dll files to a folder in your PATH. For example, the path to the MinGW bin folder is: /usr/x86_64-w64-mingw32/sys-root/mingw/bin. To read more about the PATH variable in Windows, please read the [top answer on this StackOverflow page](https://stackoverflow.com/questions/3402214/windows-7-maven-2-install). 

* Cygwin is not supported. You must install DL4J from **DOS Windows**.  

###<a name="source">Working With Source</a>

We highly recommend downloading the [Deeplearning4j JAR files from Maven Central](https://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j), rather than working with source, unless you plan on making significant commits to the project (which are always welcome, of course). To download from Maven, please follow the [instructions on the ND4J site](http://nd4j.org/getstarted.html#maven).

If you are working with source, you will need to install a [project Lombok plugin](https://projectlombok.org/download.html) for IntelliJ or Eclipse.

For a deeper dive, check out our [Github repo](https://github.com/SkymindIO/deeplearning4j/). If you want to develop for Deeplearning4j, install Github for [Mac](https://mac.github.com/) or [Windows](https://windows.github.com/). Then *git clone* the repository, and run this command for Maven:

      mvn clean install -DskipTests -Dmaven.javadoc.skip=true

If you want to run Deeplearning4j examples after installing from trunk, you should *git clone* ND4J, Canova and Deeplearning4j, in that order, and then install all from source using Maven with this command:

      mvn clean install -DskipTests -Dmaven.javadoc.skip=true

Following these steps, you should be able to run the 0.0.3.3 examples. 

###<a name="eclipse">Eclipse</a> 

After running a *git clone*, enter this command

      mvn eclipse:eclipse 
  
which will import the source and set everything up. 

### <a name="trouble">Troubleshooting</a>

* Please feel free to ask us about error messages on our [Gitter Live Chat](https://gitter.im/deeplearning4j/deeplearning4j). When you post your question, please have the following information ready (it will really speed things up!):

      * Operating System (Windows, OSX, Linux) and version 
      * Java version (7, 8) : type java -version in your terminal/CMD
      * Maven version : type mvn --version in your terminal/CMD
      * Stacktrace: Please past the error code on Gist and share the link with us: https://gist.github.com/
* If you have installed DL4J before and now see the examples throwing errors, run a git clone on [ND4J](http://nd4j.org/getstarted.html) in the same root directory as DL4J; run a clean Maven install within ND4J; install DL4J again; run a clean Maven install within DL4J, and see if that fixes things.
* When you run an example, you may get a low [f1 score](../glossary.html#f1), which is the probability that the net's classification is accurate. In this case, a low f1 doesn't indicate poor performance, because the examples train on small data sets. We gave them small data sets so they would run quickly. Because small data sets are less representative than large ones, the results they produce will vary a great deal. For example, on the minuscule example data, our deep-belief net's f1 score currently varies between 0.32 and 1.0. 
* Deeplearning4j includes an **autocomplete function**. If you are unsure which commands are available, press any letter and a drop-down list like this will appear:
![Alt text](../img/dl4j_autocomplete.png)
* Here's the Javadoc for all [Deeplearning4j's classes and methods](http://deeplearning4j.org/doc/).
* As the code base grows, installing from source requires more memory. If you encounter a Permgen error during the DL4J build, you may need to add more heap space. To do that, you'll need to find and alter your hidden .bash_profile file, which adds environmental variables to bash. To see those variables, enter *env* in the command line. To add more heap space, enter this command in your console:
      echo "export MAVEN_OPTS="-Xmx512m -XX:MaxPermSize=512m"" > ~/.bash_profile
* Older versions of Maven, such as 3.0.4, are likely to throw exceptions like a NoSuchMethodError. This can be fixed by upgrading to the latest version of Maven. 
* To compile some ND4J dependencies, you need to install some dev tools for C and C++. [Please see our ND4J guide](http://nd4j.org/getstarted.html#devtools).
* Some problems encountered using DL4J may be due to a lack of familiarity with the ideas and techniques of machine learning. We strongly encourage all Deeplearning4j users to rely on resources beyond this website to understand the fundamentals. Andrew Ng's excellent [machine-learning lectures on Coursera](https://www.coursera.org/course/ml) are a great place to start. [Geoff Hinton's neural nets course](https://www.youtube.com/watch?v=S3bx8xKpdQE), available on Youtube, is also highly instructive. While we've partially documented DL4J, many parts of the code are essentially a raw, domain-specific language for deep learning.
* The include path for [Java CPP](https://github.com/bytedeco/javacpp) doesn't always work on **Windows**. One workaround is to take the the header files from the include directory of Visual Studio, and put them in the include directory of the Java Run-Time Environment (JRE), where Java is installed. This will affect files such as standardio.h.)

### <a name="results">Reproducible Results</a>

Neural net weights are initialized randomly, which means the model begins learning from a different position in the weight space each time, which may lead it to different local optima. Users seeking reproducible results will need to use the same random weights, which they must initialize before the model is created. They can reinitialize with the same random weight with this line:

      Nd4j.getRandom().setSeed(123);

### <a name="next">Next Steps: IRIS Example & Building NNs</a>

In order to get started building neural nets, checkout the [Neural Nets Overview](http://deeplearning4j.org/neuralnet-overview.html) for more information.

Take a look at the [IRIS tutorial](../iris-flower-dataset-tutorial.html) to get running quickly and checkout [custom datasets](../customdatasets.html) for other datasets to explore.

Follow the [ND4J Getting Started](http://nd4j.org/getstarted.html) instructions to start a new project and include necessary [POM dependencies](http://nd4j.org/dependencies.html). 
