---
title: 
layout: default
---

# Installation Guide for Deeplearning4j to Run Examples on Windows 7 and above

1.)  Install Java 7 or above (https://www.java.com/de/download/)
2.)  Create an environment variable `JAVA_HOME` with the content "C:\Program Files\Java\jdk1.8.0_51"
3.)  Install [Netbeans 8.1 (currently beta)](https://netbeans.org/downloads/)
4.)  Download [Maven 3.3.3](http://ftp.fau.de/apache/maven/maven-3/3.3.3/binaries/apache-maven-3.3.3-bin.zip)
5.)  Extract Maven and add its `bin` location to the Path          
6.)  Install Tortoise Git(https://tortoisegit.org/download/)
7.)  `git clone` the following repositories to the same directory, say "deeplearning4jExamples"
	7.1.)  https://github.com/deeplearning4j/nd4j.git
	7.2.)  https://github.com/deeplearning4j/deeplearning4j.git
	7.3.)  https://github.com/deeplearning4j/Canova.git
	7.4.)  https://github.com/deeplearning4j/dl4j-0.4-examples.git
8.)  Install Visual Studio 2013 (if not freely available, download [Visual Studio Community](https://www.visualstudio.com/en-us/products/visual-studio-community-vs.aspx)
9.)  Add the Visual Studio bin directory to the Path; for example `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin` (you can take the 64 bit folder as well)
10.) To check if the Visual Studio directory was added correctly, type `cl` into the command-line tool and see if it is found.
11.) Open a command-line tool and navigate to the directory to which the repositories were downloaded.
12.) Copy each of the following lines seperately into the cmd and hit enter after each line.

		cd nd4j
		vcvars32.bat(vcvard64.bat if you chose to take the 64 bit folder into your path)
		mvn clean install -DskipTests -Dmaven.javadoc.skip=true	
		cd ../Canova
		mvn clean install -DskipTests -Dmaven.javadoc.skip=true
		cd ../deeplearning4j
		mvn clean install -DskipTests -Dmaven.javadoc.skip=true
		cd ../dl4j-0.4-examples
		mvn clean install -DskipTests -Dmaven.javadoc.skip=true

13.) Open Netbeans and import deeplearning4jExamples
14.) In Netbeans, go to Tools -> Options -> Java -> Maven and change the Maven_Home to the path of your Maven download from step 4.
15.) Congratulations, you can now run the examples.

## Running the examples on the GPU

1.) Install CUDA as described on [this site](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html#axzz3k6nvc1PO)
2.) Copy nvcc, [the Nvidia compiler](C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin) in your classpath (src\main\resources).
3.) Copy the following lines into the dependency part of the POM.xml file of your project.
    <dependency>
     <groupId>org.nd4j</groupId>
     <artifactId>nd4j-jcublas-7.0</artifactId>
     <version>${nd4j.version}</version>
    </dependency>
4.) Adapt the code for your GPU as [shown here](http://nd4j.org/dependencies.html)
5.) Recompile the module and run it.
