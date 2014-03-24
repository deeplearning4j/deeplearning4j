---
layout: default
---

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


3. Install these from source (Maven central versions coming soon).

         git clone https://github.com/agibsonccc/java-deeplearning

         cd java-deeplearning

         Use  maven: http://maven.apache.org/

         mvn clean install -DskipTests


4. Use this as a dependency in your project.

      <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-core</artifactId>
			<version>0.0.3.1</version>
		</dependency>



5. For multithreaded/clustering support, please use:

       <dependency>
			<groupId>org.deeplearning4j</groupId>
			<artifactId>deeplearning4j-scaleout-akka</artifactId>
			<version>0.0.3.1</version>
		</dependency>

From here, you can check out our [Github repo](https://github.com/agibsonccc/java-deeplearning) or access the core through [Maven](http://maven.apache.org/download.cgi).

If you're exploring deep learning, we recommend following our [MNIST tutorial](../rbm-mnist.html). If you have a clear idea of how deep learning works and what you want it to do, you can go straight to our section on [custom data sets](../customdatasets.html).