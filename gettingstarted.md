---
layout: default
---

# Getting Started

Contents

* <a href="#all">Deeplearning4j install (All OS)</a>
    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
    * <a href="#next">Running Examples</a>
    * <a href="#trouble">Troubleshooting</a>

### <a name="all">All Operating Systems</a>

* DeepLearning4J requires [Java 7](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html) or above.

###ND4J
If you aren't familiar with Nd4j, please see [here](http://nd4j.org/getstarted.html)


You can use deeplearning4j either via maven (see the readme for the dependencies) or via our [downloads](http://deeplearning4j.org/downloads.html)

You can then manually import the jar files in to [eclipse](http://stackoverflow.com/questions/3280353/how-to-import-a-jar-in-eclipse) or [intellij](http://stackoverflow.com/questions/1051640/correct-way-to-add-lib-jar-to-an-intellij-idea-project), [netbeans](http://gpraveenkumar.wordpress.com/2009/06/17/abc-to-import-a-jar-file-in-netbeans-6-5/).



### <a name="linux">Linux</a>

* Due to our reliance on Jblas for CPUs, native bindings for Blas are required.

        Fedora/RHEL
        yum -y install blas

        Ubuntu
        apt-get install libblas* (credit to @sujitpal)

* If GPUs are broken, you'll need to enter an extra command. First, find out where Cuda installs itself. It will look something like this

         /usr/local/cuda/lib64

Then enter ldconfig in the terminal, followed by the file path to link Cuda. Your command will look similar to this

         ldconfig /usr/local/cuda/lib64

If you're still unable to load Jcublas, you will need to add the parameter -D to your code (it's a JVM argument); i.e.

         java.library.path (settable via -Djava.librarypath=...) 
         // ^ for a writable directory, then 
         -D appended directly to "<OTHER ARGS>" 

If you're using IntelliJ as your IDE, this should work already. 

### <a name="osx">OSX</a>

* Install [Github](https://mac.github.com/).
* Jblas is already installed on OSX.  

### <a name="windows">Windows</a>

* Install [Git](https://windows.github.com/). Click on the Git Shell alias. Within that shell, enter the commands at the top of this page (under "all OS") to git clone the code repositories of Deeplearning4j and ND4J.

* Install [Anaconda](http://docs.continuum.io/anaconda/install.html#windows-install). If your system doesn't like the default 64-bit install, try the 32-bit offered on the same download page. (Deeplearning4j depends on Anaconda to use the graphics generator matplotlib.) 

* Install [Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/). Lapack will ask if you have Intel compilers. You do not.

* Instead, you will need to install [MinGW 32 bits](http://www.mingw.org/) (the download button is on the upper right) and then download the [Prebuilt dynamic libraries using Mingw](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw). 
* Lapack offers the alternative of [VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke). You'll also want to look at the documentation for [Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/). 

### <a name="next">Next Steps: Running Examples</a>

Follow our [**MNIST tutorial**](../mnist-tutorial.html) and try [running a few examples with our Quickstart](../quickstart.html). 

If you have a clear idea of how deep learning works and know what you want it to do, go straight to our section on [custom datasets](../customdatasets.html). 

For a deeper dive, check out our [Github repo](https://github.com/SkymindIO/deeplearning4j/) or access the core through [Maven](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j), a build automation tool used for Java projects.

### <a name="trouble">Troubleshooting</a>

* If you have installed DL4J in the past and now see the examples throwing errors, run a git clone on [ND4J](http://nd4j.org/getstarted.html) in the same root directory as DL4J; run a clean Maven install within ND4J; install DL4J again; run a clean Maven install within DL4J, and see if that fixes things.

* When you run an example, you may get a low [f1 score](../glossary.html#f1), which is the probability that the net's classification is accurate. In this case, a low f1 doesn't indicate poor performance, because the examples train on small data sets. We gave them small data sets so they would run quickly. Because small data sets are less representative than large ones, the results they produce will vary a great deal. For example, on the minuscule example data, our deep-belief net's f1 score currently varies between 0.32 and 1.0.
