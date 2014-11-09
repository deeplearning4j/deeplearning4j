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

* You can install DL4J either from source or from [Maven central](https://oss.sonatype.org/content/repositories/snapshots/org/deeplearning4j/deeplearning4j-distribution/0.0.3.2-SNAPSHOT/). Here are the **source** instructions, which require Github (Mac and Windows users, see below). 

         git clone https://github.com/SkymindIO/deeplearning4j
         git clone https://github.com/SkymindIO/dl4j-examples

###ND4J

* You'll need to install ND4J, the linear-algebra library powering DL4J, in the same root directory as DL4J itself (say, your Desktop...), like so:

         git clone https://github.com/SkymindIO/nd4j.git

### IntelliJ

* To work with DL4J code, you should download the Java IDE IntelliJ. A free, community edition is available [here](http://www.jetbrains.com/idea/download/).

* Unzip the download, move it to your applications folder, and open the application. Upon opening, you may be prompted to install a Java SE 6 runtime. If so, install it. 

* As you open IntelliJ, you will have to choose whether to create or open a project. Choose "Open Project" from the menu, and then select the working directory for Deeplearning4j. It should something like, well, "deeplearning4j". Click the open button for that folder. (It will take a while for all the dependencies to be resolved, during which time you will not be able to run your examples.) 

![Alt text](../img/open_project.png) 

* You'll need to make sure the Maven 2 Integration plugin is installed. On Macs, go to Preferences and then click on Plugins. (On Linux, you'll find the plugins in Settings.) Then choose "Browse Repositories" and search for "Maven 2 Integration." Install that plugin and restart IntelliJ. Restarting should take you back to your java-deeplearning project. 

* Click through the folder names to the examples folder -- dl4j-examples/blob/master/src/main/java/org/deeplearning4j/mnist/ -- and then right-click on the Java file you're interested in. (The MNIST dataset is where most users start.) Right click on MnistExample. In the menu that appears, look for the green arrow and choose "Run." 

![Alt text](../img/run_menu.png)

* Warning messages will appear at the top of the screen. If IntelliJ prompts you to add an SDK, choose JDK.

### Maven

* To check if you have Maven on your machine, type this in the terminal/cmd:

         mvn --version

* If you have Maven, you'll see the particular version on your computer, as well as the file path to where it lives. On a Windows PC, my file path was:

         c:\Programs\maven\bin\..

* If you don't have Maven, you can follow the installation instructions on Maven's ["getting started" page](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html). Finally, **run this command in both the ND4J and DL4J folders**:

         mvn clean install -DskipTests -Dmaven.javadoc.skip=true

* After you run a "mvn clean install", a compressed tar file with a name similar to "deeplearning4j-dist-bin.tar.gz" will be installed in the local folder (This is where you will find the jar files and it's where compiling happens.):

         */java-deeplearning/deeplearning4j-distribution/target
	
* Add the repository info below to your Project Object Model (POM) file (POM.xml files live in the root of a given directory):

         <repositories>
             <repository>
                 <id>snapshots-repo</id>
                 <url>https://oss.sonatype.org/content/repositories/snapshots</url>
                 <releases><enabled>false</enabled></releases>
                 <snapshots><enabled>true</enabled></snapshots>
             </repository>
         </repositories>

* All dependencies should be added after the tags "dependencyManagement" and "dependencies", and before they close. 
Add these POM coordinates to your project:

         <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-core</artifactId>
			<version>0.0.3.2</version>
         </dependency>
         
         //For multithreaded/clustering support, add this dependency to your POM
         
         <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-scaleout-akka</artifactId>
			<version>0.0.3.2</version>
         </dependency>
         
         //For natural-language processing, add this dependency to your POM
		
         <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-nlp</artifactId>
            <version>0.0.3.2</version>
         </dependency>

* To locally install Jcublas, which does linear algebra for GPUs, first enter these commands:

		git clone git@github.com:MysterionRise/mavenized-jcuda.git
		cd mavenized-jcuda && mvn clean install -DskipTests

  Then include linear-algebra-jcublas in your POM:

           <dependency>
             <groupId>org.deeplearning4j</groupId>
             <artifactId>linear-algebra-jcublas</artifactId>
             <version>0.0.3.2</version>
           </dependency>

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

For a deeper dive, check out our [Github repo](https://github.com/SkymindIO/deeplearning4j/) or access the core through [Maven](http://search.maven.org/#search%7Cga%7C1%7Cdeeplearning4j).

### <a name="trouble">Troubleshooting</a>

* If you have installed DL4J in the past and now see the examples throwing errors, run a git clone on [ND4J](http://nd4j.org/getstarted.html) in the same root directory as DL4J; run a clean Maven install within ND4J; install DL4J again; run a clean Maven install within DL4J, and see if that fixes things.

* When you run an example, you may get a low [f1 score](../glossary.html#f1), which is the probability that the net's classification is accurate. In this case, a low f1 doesn't indicate poor performance, because the examples train on small data sets. We gave them small data sets so they would run quickly. Because small data sets are less representative than large ones, the results they produce will vary a great deal. For example, on the minuscule example data, our deep-belief net's f1 score currently varies between 0.32 and 1.0.
