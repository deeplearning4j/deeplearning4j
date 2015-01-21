---
layout: default
---

# Getting Started

In order for you to be able to run a DL4J neural network through one of our examples, you will have to have a number of things installed in your machine. You can follow from 1 to 4 in our [ND4J.org 'getting-started' page](http://nd4j.org/getstarted.html):

1. [Java](http://nd4j.org/getstarted.html#java) 
2. [Integrated Development Environment](http://nd4j.org/getstarted.html#ide-for-java) 
3. [Maven](http://nd4j.org/getstarted.html#maven)
4. [Github](http://nd4j.org/getstarted.html#github)

And then go through the following installs:

    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
* <a href="#source">Working With Source</a>
* <a href="#eclipse">Eclipse</a>
* <a href="#trouble">Troubleshooting</a>
* <a href="#next">Next Steps</a>


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

* Install [Anaconda](http://docs.continuum.io/anaconda/install.html#windows-install). If your system doesn't like the default 64-bit install, try the 32-bit offered on the same download page. (Deeplearning4j depends on Anaconda to use the graphics generator matplotlib.) 

* Install [Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/). (Lapack will ask if you have Intel compilers. You do not.)

* To do so, you will need to install [MinGW 32 bits](http://www.mingw.org/) even if you have a 64-bit computer (the download button is on the upper right), and then download the [Prebuilt dynamic libraries using Mingw](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw). 

* Lapack offers the alternative of [VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke). You'll also want to look at the documentation for [Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/). 

###<a name="next-steps">Working With Source</a>

For a deeper dive, check out our [Github repo](https://github.com/SkymindIO/deeplearning4j/). If you want to develop for Deeplearning4j, install Github for [Mac](https://mac.github.com/) or [Windows](https://windows.github.com/). Then git clone the repository, and run this command for Maven:

      mvn clean install -DskipTests -Dmaven.javadoc.skip=true

###<a name="eclipse">Eclipse</a> 

After running a git clone, enter this command

      mvn eclipse:eclipse 
  
which will import the source and set everything up. 

### <a name="trouble">Troubleshooting</a>

* If you have installed DL4J in the past and now see the examples throwing errors, run a git clone on [ND4J](http://nd4j.org/getstarted.html) in the same root directory as DL4J; run a clean Maven install within ND4J; install DL4J again; run a clean Maven install within DL4J, and see if that fixes things.

* When you run an example, you may get a low [f1 score](../glossary.html#f1), which is the probability that the net's classification is accurate. In this case, a low f1 doesn't indicate poor performance, because the examples train on small data sets. We gave them small data sets so they would run quickly. Because small data sets are less representative than large ones, the results they produce will vary a great deal. For example, on the minuscule example data, our deep-belief net's f1 score currently varies between 0.32 and 1.0.

* Go here for a Javadoc list of [Deeplearning4j's classes and methods](http://deeplearning4j.org/doc/).

### <a name="next">Next Steps: MNIST and Running Examples</a>

Take a look at the [MNIST tutorial](../mnist-tutorial.html). If you have a clear idea of how deep learning works and know what you want it to do, go straight to our section on [custom datasets](../customdatasets.html). 
