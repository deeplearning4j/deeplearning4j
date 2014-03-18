---
layout: default
---

#Getting Started


##Setup

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


3. Install these from source (Maven central versions coming soon).


         git clone https://github.com/agibsonccc/java-deeplearning

         cd java-deeplearning

         Use [maven](http://maven.apache.org/)

         mvn clean install -DskipTests


4. Use this as a dependency in your project.

      <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-core</artifactId>
			<version>0.0.1</version>
		</dependency>



5. For multithreaded/clustering support, please use:

       <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-scaleout-akka</artifactId>
			<version>0.0.1</version>
		</dependency>


    