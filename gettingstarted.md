---
layout: default
---

*to run examples, go to our [quickstart](../quickstart.html)*
#Getting Started

1. DeepLearning4J requires [Java 7](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html) or above.

2. Due to our heavy reliance on Jblas, native bindings for Blas are required.

        Fedora/RHEL
        yum -y install blas

        Ubuntu
        apt-get install libblas* (criedit to @sujitpal)

        OSX
        Already Installed

        Windows
        See http://icl.cs.utk.edu/lapack-for-windows/lapack/

3. Install DL4J either from source or from Maven central. Below are the source instructions. Add the dependency coordinates below to your Project Object Model (POM).

         git clone https://github.com/agibsonccc/java-deeplearning
         cd java-deeplearning

4. To check if you have Maven on your machine, type this in the terminal/cmd:

         mvn --version

4.1 If you have Maven, you'll see the particular version on your computer, as well as the file path to where it lives. On a Windows PC, my file path was:

         c:\Programs\maven\bin\..

4.2 If you don't have Maven, you can follow the installation instructions on Maven's ["getting started" page](https://maven.apache.org/guides/getting-started/maven-in-five-minutes.html). Finally, run this command:

         mvn clean install -DskipTests

### dependencies

1. After you run "mvn clean", a compressed tar file with a name similar to "deeplearning4j-dist-bin.tar.gz" will be installed in the local folder (This is where you will find the jar files and it's where compiling happens.):

		*/java-deeplearning/deeplearning4j-distribution/target
	

2. Use this repo in your POM file (POM.xml files live in the root of a given directory):

         <repositories>
             <repository>
                 <id>snapshots-repo</id>
                 <url>https://oss.sonatype.org/content/repositories/snapshots</url>
                 <releases><enabled>false</enabled></releases>
                 <snapshots><enabled>true</enabled></snapshots>
             </repository>
         </repositories>

3. All dependencies should be added after the tags "dependencyManagement" and "dependencies", and before they close. Add this dependency to your POM file:

         <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-core</artifactId>
			<version>0.0.3.2-SNAPSHOT</version>
		 </dependency>

4. For multithreaded/clustering support, add this dependency to your POM file:

         <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-scaleout-akka</artifactId>
			<version>0.0.3.2-SNAPSHOT</version>
         </dependency>

5. For natural-language processing (NLP), add this dependency to your POM file:
         
         <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-scaleout-akka-word2vec</artifactId>
            <version>0.0.3.2-SNAPSHOT</version>
         </dependency>

6. To locally install Jcublas, which does linear algebra for GPUs, first enter these commands:

		git clone git@github.com:MysterionRise/mavenized-jcuda.git
		cd mavenized-jcuda && mvn clean install -DskipTests

  Then include linear-algebra-jcublas in your POM:

           <dependency>
             <groupId>org.deeplearning4j</groupId>
             <artifactId>linear-algebra-jcublas</artifactId>
             <version>0.0.3.2-SNAPSHOT</version>
           </dependency>

For the moment, the installation is throwing errors related to Jcublas. (We're working on it :) GPU integration is being completed. Steps 5 and 6 are only for building the software. 

From here, you may be interested in exploring our [Github repo](https://github.com/agibsonccc/java-deeplearning) or accessing the core through [Maven](http://maven.apache.org/download.cgi).

If you're starting to explore deep learning, we recommend following our [MNIST tutorial](../rbm-mnist-tutorial.html) or [running a few examples](../quickstart.html). If you have a clear idea of how deep learning works and know what you want it to do, go straight to our section on [custom datasets](../customdatasets.html).
