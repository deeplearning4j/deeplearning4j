---
title: Deeplearning4j With RPMs
layout: default
---

# Deeplearning4j With RPMs

Here are the steps to install Deeplearning4j with a Red Hat Package Manager (RPM):

* Set up [Spark Shell as an environment variable](http://apache-spark-user-list.1001560.n3.nabble.com/Adding-external-jar-to-spark-shell-classpath-using-ADD-JARS-td1207.html).
* Include the .repo file in this directory
        {root}/etc/yum.repos.d
* Here's what should go in the repo file:
        [dl4j.repo]
        
        name=dl4j-repo
        baseurl=http://repo.deeplearning4j.org/repo
        enabled=1

* Then enter the distros for ND4J, Canova (a vectorization lib) and DL4J. For example, you might install ND4J-jblas:

        sudo yum install nd4j-{backend}
        sudo yum install Canova-Distro
        sudo yum install Dl4j-Distro
        
That puts the JAR files on your system under /usr/local/Skymind

* The libs will be located under the project name (dl4j or nd4j or canova) 

        /usr/local/Skymind/dl4j/jcublas/lib
        /usr/local/Skymind/nd4j/jblas/lib
        
* Add each lib folder's JAR files to the classpath for Spark shell (See above). You set it in the shell script. 
* Pick a project to run. Here's an [example project](https://github.com/deeplearning4j/scala-spark-examples).
