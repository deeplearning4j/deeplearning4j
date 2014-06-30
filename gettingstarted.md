---
layout: default
---

*to run examples, go to our [quickstart](../quickstart.html)*
#getting started

1. DeepLearning4J requires [Java 7](http://www.oracle.com/technetwork/java/javase/downloads/jdk7-downloads-1880260.html) or above.

2. Due to our heavy reliance on Jblas, native bindings for Blas are required.

        Fedora/RHEL
        yum -y install blas

        Ubuntu
        apt-get install blas

        OSX
        Already Installed

        Windows
        See http://icl.cs.utk.edu/lapack-for-windows/lapack/

3. Install DL4J either from source or from Maven central. Below are the source instructions. Add the below dependency coordinates to your Project Object Model (POM).

         git clone https://github.com/agibsonccc/java-deeplearning

         cd java-deeplearning

         Use  maven: http://maven.apache.org/

         mvn clean install -DskipTests

4. Use this repo in your POM:

         <repositories>
             <repository>
                 <id>snapshots-repo</id>
                 <url>https://oss.sonatype.org/content/repositories/snapshots</url>
                 <releases><enabled>false</enabled></releases>
                 <snapshots><enabled>true</enabled></snapshots>
             </repository>
         </repositories>

5. Use this as a dependency in your project:

         <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-core</artifactId>
			<version>0.0.3.2-SNAPSHOT</version>
		 </dependency>

6. For multithreaded/clustering support, please use:

         <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-scaleout-akka</artifactId>
			<version>0.0.3.2-SNAPSHOT</version>
		</dependency>

7. For natural-language processing (NLP), use:
         
         <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-scaleout-akka-word2vec</artifactId>
            <version>0.0.3.2-SNAPSHOT</version>
         </dependency>

From here, you may be interested in exploring our [Github repo](https://github.com/agibsonccc/java-deeplearning) or accessing the core through [Maven](http://maven.apache.org/download.cgi).

If you're starting to explore deep learning, we recommend following our [MNIST tutorial](../rbm-mnist-tutorial.html) or [running a few examples](../quickstart.html). If you have a clear idea of how deep learning works and know what you want it to do, go straight to our section on [custom datasets](../customdatasets.html).
