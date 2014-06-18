---
layout: default
---

*to run examples, go to our [quickstart](../quickstart.html)*
#getting started

1. DeepLearning4J requires Java 7 or above.

2. Due to our heavy reliance on Jblas, native bindings for Blas are required.

        Fedora/RHEL
        yum -y install blas

        Ubuntu
        apt-get install blas

        OSX
        Already Installed

        Windows
        See http://icl.cs.utk.edu/lapack-for-windows/lapack/

3. Install either from source or Maven central. Below are the source instructions. Add the below dependency coordinates in to your pom, otherwise.

         git clone https://github.com/agibsonccc/java-deeplearning

         cd java-deeplearning

         Use  maven: http://maven.apache.org/

         mvn clean install -DskipTests

4. Use this repo in your pom:

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

7. For NLP, use:
         
         <dependency>
            <groupId>org.deeplearning4j</groupId>
            <artifactId>deeplearning4j-scaleout-akka-word2vec</artifactId>
            <version>0.0.3.2-SNAPSHOT</version>
         </dependency>

From here, you can check out our [Github repo](https://github.com/agibsonccc/java-deeplearning) or access the core through [Maven](http://maven.apache.org/download.cgi).

If you're exploring deep learning, we recommend following our [MNIST tutorial](../rbm-mnist-tutorial.html). If you have a clear idea of how deep learning works and what you want it to do, you can go straight to our section on [custom data sets](../customdatasets.html).
