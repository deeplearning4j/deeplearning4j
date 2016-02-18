---
layout: default
title: Getting Started on Linux
---

# Getting Started on Linux

* To run examples, go to our [quickstart](../quickstart.html).

* DeepLearning4J requires [Java 7](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html) or above.

* Due to our reliance on Jblas for CPUs, native bindings for Blas are required.

        Fedora/RHEL
        yum -y install blas
        
        Ubuntu
        apt-get install libblas* (credit to @sujitpal)

* You can install DL4J either from source or from Maven central. Here are the **source** instructions. 

         git clone https://github.com/agibsonccc/java-deeplearning
         cd java-deeplearning

### IntelliJ

* To work with DL4J code, you should download the Java IDE IntelliJ. A [free, community edition](http://www.jetbrains.com/idea/download/) is available at:

         http://www.jetbrains.com/idea/download/

* Unzip the download, move it to your applications folder, and open the application. Upon opening, you may be prompted to install a Java SE 6 runtime. If so, install it. 

* As you open IntelliJ, you will have to choose whether to create or open a project. Choose "Open Project" from the menu, and then select the working directory for Deeplearning4j. Mine was "java-deeplearning". Click the open button for that folder. (It will take a while for all the dependencies to be resolved, during which time you will not be able to run your examples.)

![Alt text](../img/open_project.png) 

* You'll need to make sure the Maven 2 Integration plugin is installed. On Macs, go to Preferences and then click on Plugins. (On Linux, you'll find the plugins in Settings.) Then choose "Browse Repositories" and search for "Maven 2 Integration." Install that plugin and restart IntelliJ. Restarting should take you back to your java-deeplearning project. 

* Click through the folder names to the examples folder -- java-deeplearning/deeplearning4j-examples/src/main/java/org/deeplearning4j/example/ -- and then right-click on the dataset you're interested in. (MNIST is where most users start.) There, you will find a number of nets that will run on MNIST. Right click on RBMMnistExample. In the menu that appears, look for the green arrow and choose "Run." 

![Alt text](../img/run_menu.png)

* Warning messages will appear at the top of the screen. If IntelliJ prompts you to add an SDK, choose JDK.

### Maven

* To check if you have Maven on your machine, type this in the terminal/cmd:

         mvn --version

* If you have Maven, you'll see the particular version on your computer, as well as the file path to where it lives. On a Windows PC, my file path was:

         c:\Programs\maven\bin\..

* If you don't have Maven, you can follow the installation instructions on Maven's ["getting started" page](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html). Finally, run this command:

         mvn clean install -DskipTests

* After you run "mvn clean", a compressed tar file with a name similar to "deeplearning4j-dist-bin.tar.gz" will be installed in the local folder (This is where you will find the jar files and it's where compiling happens.):

		*/java-deeplearning/deeplearning4j-distribution/target
	
* Add the coordinates below to your Project Object Model (POM) file (POM.xml files live in the root of a given directory):

         <repositories>
             <repository>
                 <id>snapshots-repo</id>
                 <url>https://oss.sonatype.org/content/repositories/snapshots</url>
                 <releases><enabled>false</enabled></releases>
                 <snapshots><enabled>true</enabled></snapshots>
             </repository>
         </repositories>

* All dependencies should be added after the tags "dependencyManagement" and "dependencies", and before they close. Add this dependency to your POM file:

         <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-core</artifactId>
			<version>0.0.3.2-SNAPSHOT</version>
		 </dependency>

* For multithreaded/clustering support, add this dependency to your POM file:

         <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-scaleout-akka</artifactId>
			<version>0.0.3.2-SNAPSHOT</version>
         </dependency>

* For natural-language processing (NLP), add this dependency to your POM file:
         
         <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-scaleout-akka-word2vec</artifactId>
            <version>0.0.3.2-SNAPSHOT</version>
         </dependency>

* To locally install Jcublas, which does linear algebra for GPUs, first enter these commands:

		git clone git@github.com:MysterionRise/mavenized-jcuda.git
		cd mavenized-jcuda && mvn clean install -DskipTests

  Then include linear-algebra-jcublas in your POM:

           <dependency>
             <groupId>org.deeplearning4j</groupId>
             <artifactId>linear-algebra-jcublas</artifactId>
             <version>0.0.3.2-SNAPSHOT</version>
           </dependency>

**Next step**: Follow our [**MNIST tutorial**](../rbm-mnist-tutorial.html) and [running a few examples](../quickstart.html). 

If you have a clear idea of how deep learning works and know what you want it to do, go straight to our section on [custom datasets](../customdatasets.html).

For a deeper dive, check out our [Github repo](https://github.com/agibsonccc/java-deeplearning) or access the core through [Maven](http://maven.apache.org/download.cgi).
