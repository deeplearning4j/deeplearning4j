---
layout: default
---

# Getting Started

Contents

* <a href="#quickstart">Quickstart</a>
* <a href="#all">Deeplearning4j install (All OS)</a>
    * <a href="#github">Github</a>
    * <a href="#ide-for-java">IDE for Java</a>
    * <a href="#maven">Maven</a>
    * <a href="#linux">Linux</a>
    * <a href="#osx">OSX</a>
    * <a href="#windows">Windows</a>
* <a href="#source">Working With Source</a>
* <a href="#eclipse">Eclipse</a>
* <a href="#trouble">Troubleshooting</a>
* <a href="#next">Next Steps</a>

## <a name="quickstart">Quickstart</a>

Our [Quickstart](../quickstart.html) shows you how to run your first examples. 

## <a name="all">Full Install: All OS</a>

DeepLearning4J requires [Java 7](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html) or above.

## ND4J: Numpy for the JVM

[ND4J is the Java scientific computing engine](http://nd4j.org/) powering our matrix manipulations.The ND4J getting started page is [here](http://nd4j.org/getstarted.html), and you need to install it to run DL4J. (It's also fun in and of itself...)

## <a id="github">Github</a>

* Github is **only necessary** to run DL4J examples from the Quickstart, or to help develop the framework by working on the source code. It is not necessary to install Deeplearning4j and use its neural nets, so if you do not plan to help us develop DL4J, you may not need it. In that case, proceed to the IDE. 
* Download Github for [Mac](https://mac.github.com/), [Windows](https://windows.github.com/), etc. Then enter this command into your terminal (Mac) or Git Shell (Windows):

      git clone https://github.com/SkymindIO/deeplearning4j

## <a id="ide-for-java">IDE for Java</a>

### What it is
An Integrated Development Environment ([IDE](http://encyclopedia.thefreedictionary.com/integrated+development+environment)) will allow you to edit the source code, debug it and build it with a few clicks. The ones suggested here will use your installed version of Java, will talk with GitHub and Maven, which will take care of the dependencies for you. Visit our [dependencies](dependencies.html) page to know how to 'easily' change the dependencies later on.

### Why you need it
You want to set up a hassle-free development environment so that you only worry about your code. IDEs typically come with Maven support, but we prefer you to install [Maven](#3-maven) so you can run commands directly as instructed previously.

### Is it already installed?
Just check your installed programs.

### Installation
We recommend installing [IntelliJ](https://www.jetbrains.com/idea/download/). You will be perfectly fine with the free community edition.

These are some equivalent IDEs: [Eclipse](http://books.sonatype.com/m2eclipse-book/reference/creating-sect-importing-projects.html) or [Netbeans](http://wiki.netbeans.org/MavenBestPractices).

## <a id="maven">3. Maven</a>

### What it is
Maven is an automated build tool for Java projects (among other [things](http://maven.apache.org/what-is-maven.html)), that basically locates the latest version of the libraries (Deeplearning4j .jar files), and downloads them automatically to your computer.

### Why you need it
Maven will allow you to install both ND4J and Deeplearning4j projects easily. It works well with Integrated Development Environments ([IDE](#4-ide-for-java)) such as IntelliJ.

(If you really know what you are doing, and do not want to install Maven, you can find the .jar files in our [downloads](downloads.html) page. For an expert user it might be faster, but also more complicated due to dependencies.)

### Is it already installed?
To see if Maven is installed in your machine, enter the following into the command line:

		mvn --version

### Installation
Instructions to install Maven are [here](https://maven.apache.org/download.cgi). Download the compressed file containing Maven's latest stable version.

![Alt text](../img/maven_downloads.png) 

Lower on the same Web page, follow the instructions that pertain to your operating system; e.g. *"Unix-based Operating Systems (Linux, Solaris and Mac OS X)."* They look like this:

![Alt text](../img/maven_OS_instructions.png) 

* Now, using your IDE, create a new project:

![Alt text](../img/new_maven_project.png) 

The images below will step you through the windows of the IntelliJ New Project Wizard using Maven. First you name your group and artifact

![Alt text](../img/maven2.png) 

Simply click "Next" on the following screen, and on the next one name your project. (May we suggest naming it Deeplearning4j? ;)

![Alt text](../img/maven4.png) 

Now you should go into your pom.xml file, within the new Deeplearning4j project in IntelliJ. It will look like this:

![Alt text](../img/pom_before.png) 

Now you need to add two dependencies: "deeplearning4j-core" and a linear-algebra backend like "nd4j-jblas". You will find both by searching for them on [search.maven.org](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j-core). When you find them, you'll want to click on the latest version. 

![Alt text](../img/search_maven_latest_version.png) 

From that screen, you want to copy the dependency information:

![Alt text](../img/latest_version_dependency.png) 

And paste it into your pom.xml, which should end up looking like this:

![Alt text](../img/pom_after.png) 

That's it. Once you've pasted the right dependencies into the pom (you may choose others, such as deeplearning4j-scaleout for distributed deep learning, or nd4j-jcublas for GPUs), you're done. You can create a Java file within IntelliJ and start using Deeplearning4j's API to start building neural nets. 

Alternatively, you can install DL4J using our [downloads](http://deeplearning4j.org/download.html). If you prefer the downloads over Maven, then you have to manually import the jar files into [Eclipse](http://stackoverflow.com/questions/3280353/how-to-import-a-jar-in-eclipse), [Intellij](http://stackoverflow.com/questions/1051640/correct-way-to-add-lib-jar-to-an-intellij-idea-project) or [Netbeans](http://gpraveenkumar.wordpress.com/2009/06/17/abc-to-import-a-jar-file-in-netbeans-6-5/).

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

* The [Maven download page](http://maven.apache.org/download.cgi) has extensive instructions on how to download both Maven and Java under the "Windows section." Proper configuration entails [setting certain environment variables](http://www.computerhope.com/issues/ch000549.htm). 

* Install [Anaconda](http://docs.continuum.io/anaconda/install.html#windows-install). If your system doesn't like the default 64-bit install, try the 32-bit offered on the same download page. (Deeplearning4j depends on Anaconda to use the graphics generator matplotlib.) 

* Install [Lapack](http://icl.cs.utk.edu/lapack-for-windows/lapack/). (Lapack will ask if you have Intel compilers. You do not.)

* To do so, you will need to install [MinGW 32 bits](http://www.mingw.org/) even if you have a 64-bit computer (the download button is on the upper right), and then download the [Prebuilt dynamic libraries using Mingw](http://icl.cs.utk.edu/lapack-for-windows/lapack/#libraries_mingw). 

* Lapack offers the alternative of [VS Studio Solution](http://icl.cs.utk.edu/lapack-for-windows/lapack/#lapacke). You'll also want to look at the documentation for [Basic Linear Algebra Subprograms (BLAS)](http://www.netlib.org/blas/). 

* *For DL4J developers:* Install [Github](https://windows.github.com/). Click on the Git Shell alias. Within that shell, enter the commands below to **git clone** the code repositories of ND4J and Deeplearning4j. 

      git clone https://github.com/SkymindIO/nd4j
      git clone https://github.com/SkymindIO/deeplearning4j

###<a name="source">Working With Source</a>

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
