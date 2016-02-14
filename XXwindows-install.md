---
title: Running Examples on Windows 7 and Above
layout: default
---

# Running Examples on Windows 7 and Above

<!--Follow these steps:

* Install Java 7 or above (https://www.java.com/de/download/)
*  Create an environment variable `JAVA_HOME` with the content "C:\Program Files\Java\jdk1.8.0_51"
* Install [Netbeans 8.1 (currently beta)](https://netbeans.org/downloads/) or [IntelliJ](http://nd4j.org/getstarted.html) (recommended)
* Download [Maven 3.3.3](http://ftp.fau.de/apache/maven/maven-3/3.3.3/binaries/apache-maven-3.3.3-bin.zip)
* Extract Maven and add its `bin` location to the Path          
* Install Tortoise Git(https://tortoisegit.org/download/)
* `git clone` the following repositories to the same directory, say "deeplearning4jExamples"

			https://github.com/deeplearning4j/nd4j.git
			https://github.com/deeplearning4j/deeplearning4j.git
			https://github.com/deeplearning4j/Canova.git
			https://github.com/deeplearning4j/dl4j-0.4-examples.git

* Install Visual Studio 2013 (if not freely available, download [Visual Studio Community](https://www.visualstudio.com/en-us/products/visual-studio-community-vs.aspx)
* Add the Visual Studio bin directory to the Path; for example `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin` (you can take the 64 bit folder as well)
* To check if the Visual Studio directory was added correctly, type `cl` into the command-line tool and see if it is found.
* Open a command-line tool and navigate to the directory to which the repositories were downloaded.
* Copy each of the following lines seperately into the cmd and hit enter after each line.

		cd nd4j
		vcvars32.bat(vcvars64.bat if you chose to take the 64 bit folder into your path)
		mvn clean install -DskipTests -Dmaven.javadoc.skip=true	
		cd ../Canova
		mvn clean install -DskipTests -Dmaven.javadoc.skip=true
		cd ../deeplearning4j
		mvn clean install -DskipTests -Dmaven.javadoc.skip=true
		cd ../dl4j-0.4-examples
		mvn clean install -DskipTests -Dmaven.javadoc.skip=true

* Open Netbeans and import "deeplearning4jExamples"
* In Netbeans, go to Tools -> Options -> Java -> Maven and change the Maven_Home to the path of your Maven download from step 4.
* Congratulations, you can now run the examples.

## Running the examples on a GPU

* WARNING: CUDA 7.5 and 7 may not work on Windows 10. 
* Install CUDA as described on [this site](http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html#axzz3k6nvc1PO)
* Copy nvcc, [the Nvidia compiler](C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\bin) in your classpath (src\main\resources).
* Copy the following lines into the dependency part of the POM.xml file of your project.

		    <dependency>
		     <groupId>org.nd4j</groupId>
		     <artifactId>nd4j-jcublas-7.0</artifactId>
		     <version>${nd4j.version}</version>
		    </dependency>
    
* Adapt the code for your GPU as [shown here](http://nd4j.org/dependencies.html)
* Recompile the module and run it.

(*Special thanks to user @Imdrail for help with these instructions.*)
